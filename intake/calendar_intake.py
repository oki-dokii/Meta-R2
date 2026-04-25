"""
calendar_intake.py — Extract life-state signals from Google Calendar.

Real OAuth flow mirrors gmail_intake.py. Falls back to demo_signals.json
automatically when credentials.json is absent (hackathon demo mode).

SETUP (real mode):
1. Enable Google Calendar API in console.cloud.google.com
2. Download credentials.json to the project root
3. pip install google-auth google-auth-oauthlib google-api-python-client
"""

import os
import json
from datetime import datetime, timedelta, timezone

# Monkeypatch for Python 3.9 compatibility with modern google-auth
try:
    import importlib.metadata as metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        metadata = None

if metadata and not hasattr(metadata, 'packages_distributions'):
    metadata.packages_distributions = lambda: {}

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
_DEMO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'demo_signals.json')


class CalendarIntake:
    # ── Real OAuth path ──────────────────────────────────────────────────

    def authenticate(self):
        """Return an authenticated Calendar API service. Raises if credentials missing."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        token_file = 'calendar_token.json'
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    raise FileNotFoundError("credentials.json missing.")
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_file, 'w') as f:
                f.write(creds.to_json())

        return build('calendar', 'v3', credentials=creds)

    def extract_signals(self, service, days: int = 7) -> dict:
        """Pull real calendar data and return structured signals."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy='startTime',
            maxResults=100,
        ).execute()
        events = events_result.get('items', [])

        total_minutes = 0
        back_to_back = 0
        personal = 0
        deadlines = []
        prev_end = None

        for ev in events:
            start_str = ev.get('start', {}).get('dateTime') or ev.get('start', {}).get('date')
            end_str = ev.get('end', {}).get('dateTime') or ev.get('end', {}).get('date')
            if not start_str or not end_str:
                continue
            try:
                s = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                e = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            except ValueError:
                continue

            duration = (e - s).total_seconds() / 60
            total_minutes += duration

            if prev_end and (s - prev_end).total_seconds() < 600:
                back_to_back += 1
            prev_end = e

            title = ev.get('summary', '').lower()
            if any(w in title for w in ('personal', 'gym', 'family', 'date', 'birthday', 'doctor')):
                personal += 1

            importance = ev.get('colorId')
            if importance in ('11', '4') or any(w in title for w in ('deadline', 'submit', 'launch', 'board', 'review')):
                deadlines.append({
                    "title": ev.get('summary', 'Untitled'),
                    "due_in_hours": round((s - now).total_seconds() / 3600),
                    "priority": "critical" if importance == '11' else "high",
                })

        working_minutes = days * 8 * 60
        occupancy = min(100, round(total_minutes / working_minutes * 100))
        avg_meeting_h = round(total_minutes / 60 / days, 1)
        focus_blocks = max(0, days - back_to_back - 1)

        return {
            "week_occupancy_pct": occupancy,
            "avg_meeting_hours_per_day": avg_meeting_h,
            "back_to_back_blocks": back_to_back,
            "focus_blocks_count": focus_blocks,
            "personal_events_this_week": personal,
            "upcoming_deadlines": deadlines[:3],
            "summary": (
                f"{occupancy}% of working hours booked. "
                f"{avg_meeting_h}h meetings/day. "
                f"{back_to_back} back-to-back chains. "
                f"{len(deadlines)} deadlines upcoming."
            ),
        }

    def to_life_metrics(self, signals: dict) -> dict:
        """Map calendar signals to LifeMetrics deltas."""
        occ = signals.get('week_occupancy_pct', 50)
        btb = signals.get('back_to_back_blocks', 0)
        focus = signals.get('focus_blocks_count', 3)
        return {
            "time.free_hours_per_week": -((occ - 50) / 5),
            "time.schedule_control": -(occ / 10),
            "mental_wellbeing.stress_level": (occ / 10) + (btb * 2),
            "mental_wellbeing.focus_quality": focus * 5 - 10,
            "career.workload": (occ - 50) / 2,
        }

    # ── Demo fallback ────────────────────────────────────────────────────

    @staticmethod
    def demo_signals() -> dict:
        with open(_DEMO_PATH) as f:
            return json.load(f)['calendar']

    @staticmethod
    def demo_life_metrics() -> dict:
        with open(_DEMO_PATH) as f:
            d = json.load(f)
        return {k: v for k, v in d['derived_metric_deltas'].items()
                if k.startswith('time.') or k.startswith('career.')}

    # ── Unified entry point ──────────────────────────────────────────────

    def sync(self) -> tuple[dict, dict, bool]:
        """
        Returns (signals, metric_deltas, is_demo).
        Tries real OAuth first; silently falls back to demo on any failure.
        """
        try:
            svc = self.authenticate()
            sigs = self.extract_signals(svc)
            return sigs, self.to_life_metrics(sigs), False
        except Exception:
            return self.demo_signals(), self.demo_life_metrics(), True
