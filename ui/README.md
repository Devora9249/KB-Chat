# UI – שלד Chainlit (Sprint 1)

ממשק צ'אט בסיסי ל-Knowledge Center Assistant. כרגע הוא מחזיר **תשובה מזויפת (Fake)**, כדי שאפשר יהיה לפתח את ה-UI בלי לחכות ל-API האמיתי.

## הרצה

מתוך התיקייה `ui/` (לא מהשורש):

```bash
cd ui
uv run --with chainlit chainlit run app.py
```

הדפדפן ייפתח בכתובת http://localhost:8000. אם הוא לא נפתח, פתחו את הכתובת ידנית. לעצירה: `Ctrl+C`.

> למה מתוך `ui/`? בהרצה הראשונה Chainlit יוצר בתיקייה הנוכחית את `.chainlit/` (הגדרות ותרגומים) ואת `chainlit.md` (מסך הפתיחה). כשמריצים מתוך `ui/`, הקבצים האלה נשארים בתוך התיקייה של ה-UI.

> הערה: Chainlit עדיין לא מופיע ב-`pyproject.toml` (ממתין לאישור של דבורה), ולכן מריצים אותו עם `--with chainlit`. כך uv מתקין אותו בסביבה זמנית בלבד.

## מבנה

| קובץ | תפקיד |
|---|---|
| `app.py` | אפליקציית Chainlit: מקבלת שאלה ומציגה תשובה עם קישורים למקורות |
| `fake_api.py` | פונקציית `ask(question) -> Answer` שמחזירה תשובה קבועה |
| `models.py` | מודלים מקומיים `Answer` ו-`Citation` (זמניים, עד שיוסכם החוזה עם לאה) |

## מעבר ל-API האמיתי

כשה-API `/v1/ask` של לאה יהיה מוכן, כותבים client עם אותה חתימה (`async def ask(question: str) -> Answer`), ומחליפים שורה אחת ב-`app.py`:

```python
from fake_api import ask  # ← להחליף ל-client האמיתי
```
