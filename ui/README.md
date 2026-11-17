# UI – Chainlit

ממשק צ'אט ל-Knowledge Center Assistant. כברירת מחדל הוא מתחבר ל-API האמיתי של לאה (`POST /v1/ask`), ואפשר להריץ אותו גם במצב אופליין עם תשובות מזויפות (Fake).

## הרצה: API + UI בשני טרמינלים

Chainlit ו-uvicorn משתמשים שניהם כברירת מחדל בפורט 8000, ולכן:
- **API** על פורט **8000**
- **UI** על פורט **8080**

**טרמינל 1 – ה-API (מהשורש של הריפו):**

```bash
uv run uvicorn kc.api.app:app --reload --port 8000
```

בדיקה: http://127.0.0.1:8000/health צריך להחזיר `{"status":"ok"}`. התיעוד האינטראקטיבי נמצא ב-http://127.0.0.1:8000/docs.

**טרמינל 2 – ה-UI (מתוך `ui/`):**

```bash
cd ui
uv run --with chainlit chainlit run app.py --port 8080
```

פותחים את http://localhost:8080. לעצירה: `Ctrl+C` בכל טרמינל.

> למה מתוך `ui/`? בהרצה הראשונה Chainlit יוצר בתיקייה הנוכחית את `.chainlit/` (הגדרות ותרגומים) ואת `chainlit.md` (מסך הפתיחה). כשמריצים מתוך `ui/`, הקבצים האלה נשארים בתוך התיקייה של ה-UI.

> הערה: Chainlit עדיין לא מופיע ב-`pyproject.toml` (ממתין לאישור של דבורה), ולכן מריצים אותו עם `--with chainlit`. כך uv מתקין אותו בסביבה זמנית בלבד.

### מצב אופליין (בלי API)

```bash
# bash
cd ui
KC_UI_BACKEND=fake uv run --with chainlit chainlit run app.py --port 8080
```

```powershell
# PowerShell
cd ui
$env:KC_UI_BACKEND = "fake"; uv run --with chainlit chainlit run app.py --port 8080
```

## משתני סביבה

| משתנה | ברירת מחדל | תפקיד |
|---|---|---|
| `KC_UI_BACKEND` | `api` | `api` – פנייה ל-`/v1/ask` של לאה. `fake` – תשובות מזויפות מקומיות (`fake_api.py`), בלי רשת. ערך אחר עוצר את ה-UI כבר בעלייה עם הודעת שגיאה ברורה. |
| `KC_API_BASE_URL` | `http://127.0.0.1:8000` | הכתובת של ה-API. בתוך docker compose זה יהיה `http://api:8000`. רלוונטי רק כש-`KC_UI_BACKEND=api`. |

## סטטוסים של תשובה

ה-UI מציג כל תשובה לפי השדה `status` של `contracts.generation.Answer`:

| status | מה מוצג |
|---|---|
| `answered` | טקסט התשובה, ואחריו **Sources:** – רשימה ממוספרת של קישורים (המספר = `citation.id`, תואם ל-`[n]` בטקסט) |
| `not_found` | טקסט בלבד ("לא מצאתי") |
| `needs_clarification` | טקסט בלבד – שאלת הבהרה למשתמש |

כל עוד ה-API עובד עם ה-`FakeAnswerer` של לאה (וגם במצב `fake`), אפשר לקבל כל סטטוס כך:
- שאלה שמכילה את המילה `unknown` → `not_found`
- שאלה של פחות מ-3 מילים → `needs_clarification`
- כל שאלה אחרת → `answered`

## שגיאות

אם ה-API לא זמין, מחזיר שגיאה או תשובה לא תקינה, ה-UI לא קורס: הוא מציג הודעה עם ⚠️ (למשל "Cannot reach the assistant API at http://127.0.0.1:8000. Is it running?"), והפרטים הטכניים נכתבים ללוג.

## מבנה

| קובץ | תפקיד |
|---|---|
| `app.py` | אפליקציית Chainlit: בוחרת backend לפי `KC_UI_BACKEND` ומציגה את התשובה לפי הסטטוס |
| `api_client.py` | `ask(question) -> Answer` – client ל-`POST /v1/ask` עם httpx; כל כשל הופך ל-`ApiError` עם הודעה למשתמש |
| `fake_api.py` | `ask(question) -> Answer` – תשובות מזויפות באותו מבנה, לעבודה אופליין |

המודלים `Answer` ו-`Citation` מגיעים מ-`contracts.generation` (החוזה של לאה), לא מקובץ מקומי.
