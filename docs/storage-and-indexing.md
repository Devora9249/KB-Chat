# Storage & Indexing

אחראית: Raizy. קוד: `src/kc/db/`, `src/kc/indexing/`, `contracts/storage.py`. טסטים: `tests/db/`, `tests/indexing/`.

המסמך מסביר מה התחום עושה, מה החוזים שלו מבטיחים, ואיך עובדים איתו. הוא כתוב בשביל כל הצוות, ובמיוחד בשביל מי שמשתמשת בתחום בלי לפתח בו.

## 0. בקצרה: מה כבר ב-main ומה בקרוב

| נושא | מצב |
|---|---|
| החוזים `Embedder` ו-`SearchIndex` (`contracts/storage.py`) | ב-main |
| `IndexWriter` ו-`IndexStore` (זמניים, `_provisional_contracts.py`) | ב-main |
| `FakeEmbedder` ו-`InMemoryIndexStore` (לטסטים ולפיתוח) | ב-main |
| `index_documents` (צינור האינדוקס) | ב-main |
| מיגרציות 0001 ו-0002 (pgvector, `embedding_config`) וההגנה על החלפת מודל | ב-main |
| `PgIndexStore`, ומיגרציות 0003 ו-0004 (`documents`, `chunks`, HNSW, GIN) | **בקרוב, PR #14** |
| הגדרות ה-pool של המסד (`DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`) | **בקרוב, PR #13** |
| `BgeM3Embedder` ו-`E5Embedder` (מודלים מקומיים), `token_limits` וסקריפט המדידה | **בקרוב, PR #19** |
| כלי אינדוקס שאפשר להריץ משורת הפקודה | **בקרוב** |

---

## 1. מה התחום עושה, ואיך הוא מתחבר לאחרות

התחום הופך טקסט לוקטורים, שומר אותם במסד, ומאפשר לחפש בהם.

```
Devora (Ingestion)                Storage & Indexing                     Racheli (Search & Ranking)
Crawler -> Chunker   -- Document --> Embedder: text -> vector             HybridRetriever
                     -- Chunk ----> IndexStore: saves to Postgres <-- vector_search / text_search
                                                                               |
                                                                      Leah (API): /v1/ask, /v1/search
                                                                               |
                                                                      Tami (Evaluation): Golden Set,
                                                                      metrics, choosing the model
```

- **Devora** שולחת `Document` (דף) ורשימת `Chunk` (קטעים מהדף). אני הופכת כל chunk לוקטור ושומרת הכל.
- **Racheli** מחפשת דרך החוזה `SearchIndex` (חיפוש וקטורי וחיפוש טקסט), ומאחדת את התוצאות ב-RRF. היא מקבלת גם `Embedder`, כדי להפוך את השאלה לוקטור.
- **Leah** מחברת את כל החלקים ב-API. היא יוצרת את המימושים האמיתיים פעם אחת, בעליית השרת, ומעבירה אותם הלאה.
- **Tami** מעריכה את האיכות על ה-Golden Set. כל החלפה של מודל embeddings מתואמת איתה (סעיף 7).

אף אחת לא מייבאת את המימוש שלי ישירות, רק את החוזים מ-`contracts/`. כך מחליפים Fake במימוש אמיתי בשורה אחת (סעיף 3).

---

## 2. החוזים

חוזה הוא `Protocol` של Python: רשימה של מתודות, והבטחות על ההתנהגות שלהן. mypy בודק שכל מימוש עומד בו.

### `Embedder` (`contracts/storage.py`)
הופך טקסטים לוקטורים.

```python
from contracts.storage import Embedder

vectors = await embedder.embed(["How do I configure SAML?"], input_type="query")
```

- **וקטור אחד לכל טקסט, באותו סדר.** `embed([])` מחזיר `[]`.
- **נרמול:** כל וקטור באורך 1, ולכן cosine similarity שווה למכפלה הסקלרית.
- **`dimension`:** אורך כל וקטור. הוא חייב להתאים לעמודה במסד.
- **`input_type` חובה:** `"query"` לשאלה, `"document"` לטקסט שמאנדקסים. יש מודלים שזקוקים להבדל (e5 מוסיף `"query: "` או `"passage: "`), ויש שמתעלמים ממנו (bge-m3).
- **`model_name`:** מזהה את המודל **ואת הגרסה**, למשל `"BAAI/bge-m3@5617a9f"`. החלפת מודל מחייבת אינדוקס מחדש (סעיף 7).
- **חלוקה ל-batches וניסיונות חוזרים** קורים בתוך ה-Embedder. מותר לשלוח כמה טקסטים שרוצים.
- **עבודה במקביל:** מותר לקרוא לאותו מופע מכמה קריאות async במקביל. מימוש יכול להריץ אותן אחת אחרי השנייה, אבל לעולם לא יערבב בין התוצאות.

### `SearchIndex` (`contracts/storage.py`)
צד הקריאה של האינדקס: מה ש-Racheli צריכה.

```python
from contracts.retrieval import Filters

results = await index.vector_search(query_vector, Filters(products=["sequence"]), k=20)
results = await index.text_search("Set-CoraSeQuenceSiteAuthentication", k=20)
```

- **`k`:** לכל היותר k תוצאות. זה פרמטר keyword-only (חובה לכתוב `k=`), וברירת המחדל היא 10. `k <= 0` זורק `ValueError`.
- **סדר:** מהציון הגבוה לנמוך. בציונים שווים, לפי `chunk_id`, כך שהסדר תמיד קבוע.
- **ציונים ניתנים להשוואה רק בתוך אותה שיטה:** ב-`vector_search` זה cosine similarity, וב-`text_search` זה דירוג טקסט. לכן מאחדים לפי מיקום ברשימה (RRF), ולא לפי הציון.
- **סינון לפני k:** `filters` מופעלים **לפני** שבוחרים את k הטובים, כך ש-k סופר רק chunks שעברו את הסינון. ייתכנו פחות מ-k תוצאות.
- **`vector_search`:** הווקטור חייב להיות במימד של האינדקס, אחרת `ValueError`.
- **`text_search`:** מקבל את הטקסט של המשתמשת כמו שהוא. הפירוק שלו לביטוי חיפוש הוא באחריות האינדקס.
- **עבודה במקביל:** מופע אחד לכל האפליקציה, ובטוח לקריאות במקביל. חיפוש שרץ בזמן עדכון של מסמך רואה את הגרסה הישנה או את החדשה, אף פעם לא תערובת.
- **אין עקביות בין קריאות נפרדות:** `vector_search` ו-`text_search` באותה בקשה עלולים לראות גרסאות שונות של מסמך, אם הוא התעדכן ביניהן. כל `SearchResult` מכיל את הטקסט וה-URL שלו, ולכן עומד בפני עצמו.
- **HNSW הוא חיפוש בקירוב:** האינדקס הווקטורי (סעיף 5) מוצא את הוקטורים הקרובים מהר, אבל לא מבטיח את הקרובים ביותר בדיוק.
  - ב-`PgIndexStore` (בקרוב, PR #14), הפונקציה `vector_search_settings(k)` מגדירה לכל חיפוש:
    - `hnsw.ef_search`: כמה מועמדים לבדוק. הערך הוא `max(40, k + 10)`, עד 1000.
    - `hnsw.iterative_scan = strict_order`: אם הסינון פוסל את המועמדים הראשונים, ממשיכים לסרוק עד שיש מספיק תוצאות, בסדר מדויק.
  - את הכיוונון נעשה עם Racheli, על נתונים אמיתיים.

### `IndexWriter` ו-`IndexStore` (זמניים: `src/kc/indexing/_provisional_contracts.py`)
צד הכתיבה. הם תלויים ב-`Document` וב-`Chunk`, שייכנסו ל-`contracts/ingestion.py` ב-PR #4 של Devora. אחרי שהוא ימוזג, הם יעברו ל-`contracts/storage.py` והקובץ הזמני יימחק. מי שרק מחפשת צריכה את `SearchIndex` בלבד.

```python
await store.upsert_document(document, chunks, vectors)   # replaces ALL chunks of the document
await store.delete_document(url)                         # no-op if the URL does not exist
hashes = await store.get_document_hashes()               # {url: content_hash}, for change detection
```

- **`upsert_document`** מחליף **את כל** ה-chunks של המסמך, באופן אטומי: הכל או כלום. chunks שנעלמו בסריקה החדשה נמחקים.
  - הוא זורק `ValueError`, ולא משנה כלום, אם:
    - מספר ה-chunks שונה ממספר הוקטורים;
    - לוקטור יש מימד שגוי;
    - chunk שייך למסמך אחר;
    - יש `id` כפול (בקרוב, PR #14).
- **`IndexStore` = `SearchIndex` + `IndexWriter`.**
- **`Chunk.id`** (החלטה D12): 16 תווי hex, `sha256(url | heading_path | part)[:16]`. ה-Chunker מייצר אותו. הוא ייחודי בכל האינדקס כי ה-URL חלק מה-hash.

### שגיאות ו-timeouts (לכל החוזים)
| מה קרה | מה נזרק |
|---|---|
| קלט לא תקין (באג אצל מי שקוראת) | `ValueError`, בלי עטיפה |
| כשל של ה-Embedder (רשת, הורדה, זיכרון), אחרי הניסיונות החוזרים שלו | `EmbeddingError`, עם השגיאה המקורית ב-`__cause__` |
| כשל תשתית באינדקס (חיבור, מסד, timeout) | `IndexStoreError`, עם `__cause__`. אין ניסיונות חוזרים בתוך האינדקס: מי שקוראת מחליטה |
| ביטול (`asyncio.CancelledError`) | לא נעטף אף פעם |

- **timeout פנימי** של המימוש (הורדה, חיבור למסד, המתנה ל-pool) נעטף ב-`EmbeddingError` או ב-`IndexStoreError`.
- **timeout שמי שקוראת מגדירה** (למשל `asyncio.timeout`) מבטל את הקריאה עם `CancelledError`, שלא נעטף. לכן הוא מגיע אליה כ-`TimeoutError` מהקוד שלה.
- כל מימוש זורק תמיד חריגה אחת, אף פעם לא `ExceptionGroup`.

```python
from contracts.storage import EmbeddingError, IndexStoreError

try:
    async with asyncio.timeout(5):
        results = await index.vector_search(vector, k=10)
except TimeoutError:
    ...  # my own 5-second budget ran out
except IndexStoreError as e:
    ...  # database problem; e.__cause__ holds the original error
```

---

## 3. המימושים, ומתי להשתמש בכל אחד

| מימוש | חוזה | מתי | מצב |
|---|---|---|---|
| `FakeEmbedder` (`kc.indexing.fake_embedder`) | `Embedder` | טסטים ופיתוח. וקטורים דטרמיניסטיים מתוך hash של המילים, בלי מודל ובלי רשת. טקסטים עם מילים משותפות יוצאים קרובים, אבל **אין לו הבנה של משמעות**: מילים נרדפות, או שאלה בעברית מול מסמך באנגלית, לא יצאו קרובות | ב-main |
| `InMemoryIndexStore` (`kc.indexing.in_memory_index_store`) | `IndexStore` | טסטים ופיתוח, בלי מסד. מקיים את כל החוזה, כולל סינון ואטומיות | ב-main |
| `BgeM3Embedder` (`kc.indexing.bge_m3_embedder`) | `Embedder` | ייצור. מודל מקומי וחינמי, 1024 מימדים, עד 8192 טוקנים, כולל עברית | בקרוב, PR #19 |
| `E5Embedder` (`kc.indexing.e5_embedder`) | `Embedder` | חלופה ל-bge-m3. 1024 מימדים, **עד 512 טוקנים** | בקרוב, PR #19 |
| `PgIndexStore` (`kc.db.pg_index_store`) | `IndexStore` | ייצור. PostgreSQL עם pgvector | בקרוב, PR #14 |

ההחלטה בין bge-m3 ל-e5 תתקבל לפי ההערכה של Tami על ה-Golden Set (סעיף 8).

**החלפת Fake במימוש אמיתי** (Dependency Injection): הקוד שמשתמש במימוש מקבל אותו כפרמטר, ולכן ההחלפה היא שורה אחת במקום שיוצר אותו:

```python
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.retrieval.hybrid_retriever import HybridRetriever

embedder = FakeEmbedder(dimension=1024)
store = InMemoryIndexStore(dimension=1024, model_name=embedder.model_name)
retriever = HybridRetriever(embedder, store)

# Production (after PR #14 and #19), only these two lines change:
# embedder = BgeM3Embedder()
# store = PgIndexStore(create_session_factory(create_engine(load_db_settings())))
```

---

## 4. ה-extra של המודלים המקומיים, וטסטי המודל האמיתי (בקרוב, PR #19)

המודלים המקומיים צריכים `sentence-transformers` ו-`torch`. אלה חבילות כבדות, ולכן הן ב-**extra אופציונלי**. `uv sync` רגיל, וגם ה-CI, לא מתקינים אותן, כך שה-CI נשאר מהיר.

```powershell
uv sync --extra local-embeddings     # one-time; torch is CPU-only
```

רוב הטסטים משתמשים ב-encoder מזויף ורצים בלי ה-extra. טסטים שטוענים את המודל האמיתי מסומנים `real_model`, ורצים רק כשמבקשים במפורש:

```powershell
$env:KC_REAL_EMBEDDER = "1"; uv run pytest -m real_model tests/indexing      # PowerShell
KC_REAL_EMBEDDER=1 uv run pytest -m real_model tests/indexing                # bash
```

- **בהרצה הראשונה כל מודל יורד מ-Hugging Face**, כ-2.3GB. אחר כך הוא נטען מה-cache המקומי.
- הטסטים בודקים מימד 1024 ונרמול, ששאלה בעברית מוצאת את המסמך הנכון באנגלית, ו-30 קריאות במקביל.
- **כל מודל נטען מגרסה נעולה** (`MODEL_REVISION`), כך שבכל מחשב יש בדיוק אותם משקלים.
- **טעינה עצלה:** המודל נטען בקריאה הראשונה ל-`embed`, ולא כשיוצרים את האובייקט. בשרת כדאי לחמם אותו בעלייה, בקריאה אחת ל-`embed`.

---

## 5. המסד

PostgreSQL 16 עם ההרחבה **pgvector**: וקטורים, metadata וחיפוש טקסט, הכל במסד אחד.

### הטבלאות
| טבלה | מה יש בה | מצב |
|---|---|---|
| `embedding_config` | שורה אחת: איזה מודל (`model_name`, `dimension`) בנה את האינדקס. ההגנה של סעיף 7 | ב-main (0002) |
| `documents` | שורה לכל דף: `url` (מפתח), `title`, `breadcrumb`, `product`, `version`, `doc_type`, `content_md`, `content_hash`, `indexed_at` | בקרוב, PR #14 (0003) |
| `chunks` | שורה לכל chunk: `id` (מפתח), `document_url`, `text`, `heading_path`, `product`, `version`, `doc_type`, `tsv`, `embedding` | בקרוב, PR #14 (0003, 0004) |

- `product`, `version` ו-`doc_type` מועתקים מהמסמך לכל chunk, כדי שהסינון לא יצטרך JOIN.
- מחיקת מסמך מוחקת גם את ה-chunks שלו (`ON DELETE CASCADE`).
- `chunks.id` הוא המפתח הראשי. ה-Chunker כבר מבטיח ייחודיות (D12), והמפתח הוא רשת ביטחון.

### האינדקסים
- **HNSW על `embedding`**, עם `vector_cosine_ops`. זה גרף שמאפשר למצוא וקטורים קרובים בלי לעבור על כל השורות. הוא חיפוש בקירוב (סעיף 2).
- **GIN על `tsv`.** `tsv` הוא ייצוג של הטקסט כרשימת מילים, ו-Postgres מחשב אותו בעצמו. משתמשים במילון `simple`, בלי גזירה ובלי מילות עצירה, כדי ששמות טכניים מדויקים כמו `Set-CoraSeQuenceSiteAuthentication` יימצאו.
- **אינדקסים רגילים** על `product`, `version`, `doc_type` (לסינון) ועל `document_url` (להחלפה ולמחיקה של מסמך).

### הרצת מיגרציות
רק Raizy יוצרת מיגרציות. מי שצריכה שינוי במסד פותחת issue.

```powershell
# DATABASE_URL in .env, e.g. postgresql+psycopg://user:password@localhost:5432/kc
uv run alembic -c src/kc/db/alembic.ini upgrade head              # apply all migrations
uv run alembic -c src/kc/db/alembic.ini upgrade head --sql        # show the SQL only; does not connect
```

גם עם `--sql` צריך ש-`DATABASE_URL` יהיה מוגדר, אבל הפקודה לא מתחברת אליו, ולכן אפשר להשתמש בכתובת דמה.

- `DATABASE_URL` חייב להתחיל ב-`postgresql+psycopg://`. הסיסמה לא מודפסת בשגיאות.
- **ב-Windows:** הדרייבר psycopg במצב async לא עובד עם ה-event loop הרגיל. בקוד async מקומי מריצים כך: `asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)`. ב-Linux (Docker, CI) אין בעיה.
- **טסטי האינטגרציה** (`tests/db/`, וטסטי החוזה של `PgIndexStore`) רצים רק כש-`DATABASE_URL` מוגדר, כמו ב-CI. בלעדיו הם מדלגים. הם מריצים מיגרציות ומוחקים נתונים **רק במסד ששמו מכיל `test`**, כדי שלא יפגעו במסד אמיתי בטעות.

### הגדרות ה-pool של החיבורים (בקרוב, PR #13)
ה-engine מחזיק pool: מאגר של חיבורים פתוחים למסד, שמשתמשים בהם שוב ושוב.

| משתנה | מה הוא עושה | ברירת מחדל |
|---|---|---|
| `DB_POOL_SIZE` | כמה חיבורים נשארים פתוחים קבוע | 5 |
| `DB_MAX_OVERFLOW` | כמה חיבורים זמניים מותר לפתוח מעבר ל-`DB_POOL_SIZE` בזמן עומס. לא יכול להיות שלילי (בלי "בלי הגבלה"), כדי לא להציף את המסד | 10 |
| `DB_POOL_TIMEOUT` | כמה שניות קריאה מחכה לחיבור פנוי לפני שהיא נכשלת. ב-`PgIndexStore` זה הופך ל-`IndexStoreError` | 30 |

- ברירות המחדל הן של SQLAlchemy, כך שבלי הגדרה ההתנהגות לא משתנה.
- לכל היותר `DB_POOL_SIZE + DB_MAX_OVERFLOW` חיבורים בבת אחת, כלומר 15 בברירת המחדל.
- **`HybridRetriever` מריץ את `vector_search` ואת `text_search` במקביל**, ולכן כל חיפוש תופס עד **2 חיבורים**. בברירת המחדל זה בערך 7 חיפושים במקביל לפני שמתחילים לחכות.
- את הערכים נכוון אחרי מדידה מול עומס אמיתי.

---

## 6. איך מאנדקסים מסמכים

`index_documents` (`kc.indexing.pipeline`) לוקח זוגות של (מסמך, chunks), מחשב וקטורים ושומר.

```python
from kc.indexing._provisional_contracts import Chunk, Document
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.pipeline import index_documents

embedder = FakeEmbedder(dimension=1024)
store = InMemoryIndexStore(dimension=1024, model_name=embedder.model_name)

report = await index_documents([(document, chunks)], embedder=embedder, store=store)
print(report.indexed)   # URLs that were indexed
print(report.failed)    # {url: "EmbeddingError: ..."} for documents that failed
```

לכל מסמך מתבצעים שלושה צעדים:
1. **בדיקה זולה:** כל chunk שייך למסמך. היא מתבצעת לפני שמשלמים על embeddings.
2. **embedding** של כל ה-chunks, עם `input_type="document"`.
3. **`upsert_document`:** החלפה אטומית של המסמך באינדקס.

- **מסמך שנכשל לא עוצר את הריצה.** אם ה-embedding נכשל, צעד 3 לא מתבצע, ו**הגרסה הקודמת של המסמך נשארת באינדקס**. עדיף לענות מגרסה קצת ישנה מאשר לאבד דף.
- רק `EmbeddingError` ו-`ValueError` נרשמים ב-`report.failed`. כל שגיאה אחרת היא באג, והיא עוצרת את הריצה.
- **זיהוי שינויים:** `store.get_document_hashes()` מחזיר `{url: content_hash}`. מאנדקסים מחדש רק דפים שה-hash שלהם השתנה. על CPU, embedding של chunk לוקח כשנייה וחצי, כך ש-5,000 chunks לוקחים כשעתיים.

**בקרוב: כלי אינדוקס שאפשר להריץ משורת הפקודה.**

---

## 7. החלפת מודל embeddings

### למה צריך אינדוקס מחדש
וקטורים של מודלים שונים חיים ב"מרחבים" שונים. וקטור של שאלה ממודל אחד לא ניתן להשוואה לוקטורים של מסמכים ממודל אחר, **גם כשהמימד זהה** (למשל 1024 ב-bge-m3 וב-e5). התוצאות לא יהיו שגויות בצורה גלויה: הן פשוט יהיו גרועות, בלי שגיאה. גם שינוי **גרסה** של אותו מודל משנה את הוקטורים. לכן כל החלפה של מודל או גרסה מחייבת לאנדקס הכל מחדש.

### ההגנה של `embedding_config`
- בפעם הראשונה שמאנדקסים, הטבלה `embedding_config` רושמת את `model_name` ואת `dimension` של המודל.
- בכל עלייה של האפליקציה משווים את המודל המוגדר למה שרשום. אם יש הבדל, נזרק `EmbeddingConfigMismatchError` והאינדוקס לא מתחיל.
- ב-`PgIndexStore` (בקרוב, PR #14) קוראים לזה פעם אחת בעלייה:

```python
await store.verify_embedding_config(embedder.model_name, embedder.dimension)
```

- **`model_name` כולל את הגרסה** (למשל `"BAAI/bge-m3@5617a9f"`), ולכן ההגנה תופסת גם החלפת גרסה של אותו מודל.
- **מודל עם מימד אחר** (לא 1024) דורש גם מיגרציה חדשה לעמודה `embedding` ולאינדקס ה-HNSW.

### מה עושים בפועל
1. **מתאמים עם Tami:** היא משווה את המודלים על ה-Golden Set, וההחלטה מתקבלת לפי המספרים.
2. **אם בוחרים ב-e5:** הוא חותך כל טקסט שארוך מ-512 טוקנים (כולל `"passage: "`), והסוף של chunk כזה לא נכנס לוקטור בכלל. במדידה, 34 מתוך 100 chunks באורך 300 עד 600 טוקנים נחתכו. לכן **צריך לתאם עם Devora chunks של עד כ-500 טוקנים**.
3. **בודקים כמה chunks נחתכים**, עם `token_limits` (בקרוב, PR #19). הוא סופר עם הטוקנייזר של המודל עצמו, ומוריד רק את הטוקנייזר, כ-20MB:

```python
from kc.indexing.token_limits import E5_LARGE, chunks_over_limit

report = chunks_over_limit([chunk.text for chunk in chunks], E5_LARGE)
print(f"{report.over_limit} of {report.total} chunks are truncated")
```

4. **מאנדקסים הכל מחדש** למסד ריק (או אחרי מחיקת התוכן ושורת `embedding_config`).

---

## 8. סקריפט המדידה (בקרוב, PR #19)

```powershell
uv sync --extra local-embeddings
uv run python -m kc.indexing.benchmarks.embedder_benchmark --model bge-m3
uv run python -m kc.indexing.benchmarks.embedder_benchmark --model e5
```

הסקריפט מדפיס טבלת Markdown עם אותן שורות לכל מודל, כדי שאפשר יהיה להשוות:
- **הפעלה קרה:** הזמן עד שהשרת מוכן (ייבוא torch וטעינת המודל).
- **שאלה בודדת אחרי חימום:** ממוצע, חציון ו-p95, על 10 שאלות בעברית ו-10 באנגלית.
- **20 שאלות במקביל:** הקריאות עומדות בתור, ולכן האחרונה מחכה לכל השאר. זה חשוב לבחירת ה-timeout של החיפוש.
- **100 chunks** באורך 300 עד 600 טוקנים: זמן ל-chunk, כלומר כמה זמן ייקח אינדוקס מלא.
- **זיכרון בשיא:** כמה זיכרון השרת צריך.
- **chunks שחורגים מהאורך המקסימלי:** כמה מתוך ה-100 נחתכים.

התוצאות על מחשב פיתוח (CPU של 14 ליבות):

| מדידה | bge-m3 | e5-large |
|---|---|---|
| הפעלה קרה | 18.1 שניות | 26.2 שניות |
| שאלה בודדת (חציון) | 101ms | 144ms |
| 20 שאלות במקביל, עד התשובה האחרונה | 1,898ms | 4,139ms |
| chunk אחד באינדוקס | 1,831ms | 1,936ms |
| זיכרון בשיא | 2.4GB | 3.4GB |
| chunks שנחתכים (מתוך 100) | 0 | 34 |

התוצאות משתנות לפי העומס על ה-CPU, אז כדאי להריץ כמה פעמים.

---

## 9. הוספת מימוש חדש, והרצת טסטי החוזה עליו

טסטי החוזה (`tests/indexing/contract/`) בודקים את ההבטחות של סעיף 2. כל טסט רץ פעם אחת על **כל** מימוש רשום, כך שמימוש חדש מקבל את כל הבדיקות בלי לכתוב אותן מחדש.

### 1. רישום המימוש
ב-`tests/indexing/contract/conftest.py`:

```python
# Index: a factory that gets a dimension and returns a NEW, EMPTY store
IMPLEMENTATIONS: dict[str, StoreFactory] = {
    "in_memory": _in_memory,
    "my_store": lambda dimension: MyStore(dimension=dimension),
}

# Embedder
EMBEDDER_IMPLEMENTATIONS = {
    "fake": lambda: FakeEmbedder(dimension=64),
    "my_embedder": lambda: MyEmbedder(),
}
```

- מימוש שטוען מודל כבד מסמנים כ-`real_model`, כדי שירוץ רק עם `KC_REAL_EMBEDDER=1`.
- **אחרי PR #19:** כל factory ב-`EMBEDDER_IMPLEMENTATIONS` מקבל את `request` של pytest, כדי שמודלים כבדים ישתמשו במופע אחד לכל הריצה. הרשימה `REAL_MODEL_EMBEDDERS` קובעת מי מסומן `real_model`.

### 2. בדיקה סטטית
ב-`src/kc/indexing/_protocol_checks.py` מוסיפים שורה. mypy ייכשל אם המימוש לא עומד בחוזה, למשל אם חסרה מתודה או שהחתימה שגויה:

```python
_my_as_embedder: Embedder = MyEmbedder()
```

### 3. הרצה

```powershell
uv run pytest tests/indexing/contract                     # all implementations
uv run pytest tests/indexing/contract -k my_store         # only the new one
uv run mypy                                               # static check
```

### טסטים שכדאי לכתוב בנוסף
טסטי החוזה בודקים רק את מה שמשותף לכל המימושים. דברים שייחודיים למימוש, כמו עטיפת שגיאות ספציפיות או הגדרות של המודל, נבדקים בקובץ טסטים משלו.
- דוגמאות: `tests/indexing/test_in_memory_index_store.py`, ובקרוב `test_bge_m3_embedder.py` ו-`test_pg_index_store.py`.
- עדיף טסטים מהירים עם fake. מה שצריך מודל אמיתי או מסד מסמנים בהתאם (`real_model`, או דילוג כש-`DATABASE_URL` לא מוגדר).
