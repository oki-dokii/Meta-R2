"""
gmail_intake.py — Extract life-state signals from Gmail.

SETUP:
1. Same Google Cloud project as Calendar (already created)
2. Enable Gmail API in console.cloud.google.com
3. Add Gmail scope to existing credentials.json
4. pip install google-auth google-auth-oauthlib google-api-python-client
"""

import os.path
import base64
import json
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail readonly scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailIntake:
    def authenticate(self):
        """Authenticate with Gmail API, reusing token.json if possible."""
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    raise FileNotFoundError("credentials.json missing. Please download from Google Cloud Console.")
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        return build('gmail', 'v1', credentials=creds)

    def _get_headers(self, message):
        """Helper to extract common headers."""
        headers = message['payload'].get('headers', [])
        return {h['name'].lower(): h['value'] for h in headers}

    def _is_personal(self, email_addr):
        """Heuristic for personal vs work emails."""
        personal_domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'icloud.com', 'me.com']
        domain = email_addr.split('@')[-1] if '@' in email_addr else ""
        return domain in personal_domains

    def extract_relationship_signals(self, service, days=7) -> dict:
        """Fetch headers and extract relationship health signals."""
        try:
            after_date = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
            query = f'after:{after_date}'
            results = service.users().messages().list(userId='me', q=query, maxResults=100).execute()
            messages = results.get('messages', [])

            unique_senders = set()
            late_night_emails = 0
            weekend_emails = 0
            sender_counts = {}
            unanswered_threads = 0

            for msg_summary in messages:
                msg = service.users().messages().get(userId='me', id=msg_summary['id'], format='metadata', metadataHeaders=['From', 'Date']).execute()
                headers = self._get_headers(msg)
                
                sender = headers.get('from', '')
                unique_senders.add(sender)
                sender_counts[sender] = sender_counts.get(sender, 0) + 1

                # Parse date
                # Basic parsing for "Tue, 22 Apr 2026 02:36:23 +0000" or similar
                date_str = headers.get('date', '')
                try:
                    # Stripping timezone for simplicity in time/weekend check
                    clean_date = ' '.join(date_str.split(' ')[:5])
                    dt = datetime.strptime(clean_date, "%a, %d %b %Y %H:%M:%S")
                    if dt.hour >= 22 or dt.hour <= 4:
                        late_night_emails += 1
                    if dt.weekday() >= 5: # Sat or Sun
                        weekend_emails += 1
                except:
                    pass

            # Identifying "Boss" (most frequent non-personal sender)
            potential_boss = "Unknown"
            max_freq = 0
            for s, count in sender_counts.items():
                if not self._is_personal(s) and count > max_freq:
                    max_freq = count
                    potential_boss = s

            # Scores 0-10
            social_activity = min(10, len(unique_senders) / 2)
            work_pressure = min(10, max_freq)
            # Risk rises if late night work emails are high and social activity is low
            relationship_neglect_risk = min(10, (late_night_emails / 3) + (10 - social_activity) / 2)

            return {
                "social_activity": social_activity,
                "work_pressure": work_pressure,
                "relationship_neglect_risk": relationship_neglect_risk,
                "key_contacts": list(sender_counts.keys())[:5],
                "late_night_count": late_night_emails,
                "weekend_count": weekend_emails
            }
        except Exception as e:
            print(f"Gmail relationship extraction Error: {e}")
            return {"social_activity": 5, "work_pressure": 5, "relationship_neglect_risk": 5, "key_contacts": []}

    def extract_work_signals(self, service, days=7) -> dict:
        """Extract workload and work-life balance signals."""
        try:
            # Query for unread emails
            unread_results = service.users().messages().list(userId='me', q='is:unread', maxResults=50).execute()
            unread_count = len(unread_results.get('messages', []))

            # Query for emails after 6pm
            after_date = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
            overtime_results = service.users().messages().list(userId='me', q=f'after:{after_date} after:18:00', maxResults=50).execute()
            overtime_count = len(overtime_results.get('messages', []))

            email_overload = min(10, unread_count / 5)
            responsiveness = max(0, 10 - (unread_count / 10))
            work_bleeding_personal = min(10, overtime_count / 3)

            return {
                "email_overload": email_overload,
                "responsiveness": responsiveness,
                "work_bleeding_personal": work_bleeding_personal,
                "overtime_count": overtime_count,
                "unread_count": unread_count
            }
        except Exception as e:
            print(f"Gmail work extraction Error: {e}")
            return {"email_overload": 5, "responsiveness": 5, "work_bleeding_personal": 5}

    def to_life_metrics(self, rel_signals, work_signals) -> dict:
        """Map signals to LifeMetrics adjustments (deltas)."""
        return {
            "relationships.social": 40 + (rel_signals['social_activity'] * 6),
            "relationships.romantic": 100 - (rel_signals['relationship_neglect_risk'] * 7),
            "mental_wellbeing.stress_level": work_signals['email_overload'] * 3, # This is a delta
            "time.free_hours_per_week": -(work_signals['work_bleeding_personal'] * 2), # This is a delta
            "career.professional_network": 40 + (work_signals['responsiveness'] * 6)
        }

    def get_email_summary(self, rel_signals, work_signals) -> str:
        """Natural language summary of findings."""
        return (
            f"You have {work_signals.get('unread_count', 0)} unread emails. "
            f"You sent {rel_signals.get('late_night_count', 0)} emails after 10pm. "
            f"Overtime activity: {work_signals.get('overtime_count', 0)} emails after 6pm. "
            f"Social reach: {rel_signals.get('social_activity', 0)*2:.0f} unique contacts this week."
        )

def main():
    print("📧 LifeStack Gmail Intake Module")
    print("-" * 30)
    
    intake = GmailIntake()
    try:
        service = intake.authenticate()
        rel = intake.extract_relationship_signals(service)
        work = intake.extract_work_signals(service)
        
        print("\n[📊 SIGNALS]")
        print(f"  Relationship Neglect Risk: {rel['relationship_neglect_risk']:.1f}/10")
        print(f"  Work Bleeding into Life : {work['work_bleeding_personal']:.1f}/10")
        print(f"  Email Overload          : {work['email_overload']:.1f}/10")
        
        print("\n[📝 SUMMARY]")
        print(f"  {intake.get_email_summary(rel, work)}")
        
        print("\n[📈 METRIC ADJUSTMENTS]")
        deltas = intake.to_life_metrics(rel, work)
        for path, val in deltas.items():
            print(f"  {path:30}: {val:+.1f}")
            
    except Exception as e:
        print(f"\n❌ Intake failed: {e}")
        print("Note: This module requires credentials.json and a valid Google account.")

if __name__ == "__main__":
    main()
