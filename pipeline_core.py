import asyncio
import hashlib
import json

import pandas as pd
from openai import AsyncOpenAI

# ── Константи ────────────────────────────────────────────────────────────────
LLM_MODEL = 'gpt-5.4-mini'
MIN_PARA_LEN = 80
BATCH_SIZE = 20
FRAMES_SEMAPHORE = 20
TAGS_SEMAPHORE = 20

ACTION_STATUSES = ['planned', 'implemented']
URGENCY_LEVELS = ['low/medium', 'high']
BENEFICIARY_CATEGORIES = ['люди/громадяни', 'бізнес']

# ── Промпти (скопійовано з pipeline.ipynb) ───────────────────────────────────
LLM_FILTER_PROMPT = """Ти класифікуєш абзаци з публічних джерел.
Відповідай ТІЛЬКИ "YES" або "NO" для кожного абзацу.
Працюй КОНСЕРВАТИВНО: якщо є сумнів, відповідь має бути NO.

YES — лише якщо абзац МІСТИТЬ ЯВНИЙ і КОНКРЕТНИЙ внутрішньоекономічний або соціально-економічний зміст про Україну.
Потрібна пряма згадка хоча б однієї з таких тем:
- бюджет, податки, державні видатки, фінансування
- ціни на товари чи послуги, тарифи, субсидії, комунальні послуги
- зарплати, пенсії, соціальні виплати, допомога домогосподарствам
- бізнес, інвестиції, кредити, робочі місця, зайнятість, ринок праці
- промисловість, виробництво, енергетика для населення
- АКТИВНЕ відновлення або запуск інфраструктури й послуг: дороги, мости, залізничне сполучення, транспортні мережі, електрика, вода, газ, пошта, лікарні, школи
- медицина, освіта, житло, харчування, якщо йдеться саме про доступність, фінансування, вартість або державну програму

ВАЖЛИВА ЛОГІКА ДЛЯ ІНФРАСТРУКТУРИ:
- якщо щось ВІДНОВЛЮЮТЬ, ЗАПУСКАЮТЬ, ПОВЕРТАЮТЬ В РОБОТУ, ВІДКРИВАЮТЬ або РЕМОНТУЮТЬ — це YES
- якщо лише сказано, що щось ЗНИЩЕНО, ПОШКОДЖЕНО, ЗРУЙНОВАНО, без опису відновлення чи іншої економічної дії — це NO

NO — якщо абзац не містить такої прямої економічної суті, навіть якщо там є держава, суспільство чи управління.
Зокрема NO для таких випадків:
- бойові дії, зброя, фронт, обстріли, ППО, дрони, військові операції
- міжнародна дипломатія, санкції, безпека без внутрішньоекономічної конкретики
- правоохоронці, СБУ, ДБР, прокуратура, поліція, зрадники, покарання, правосуддя
- корупція, схеми, зловживання, кадри, управління державою, якщо не названо конкретний економічний об'єкт
- гуманітарна чи медична допомога без теми вартості, доступності, виплат, бюджету або державної програми
- лише руйнування, обстріли, знищення інфраструктури без відновлення, запуску або ремонту
- метафори ("ціна перемоги", "ціною життів")

ВАЖЛИВО — ЗМІШАНІ ТЕМИ:
Якщо абзац містить кілька тем одночасно (наприклад, безпека + економіка, або дипломатія + ціни),
оцінюй кожну частину НЕЗАЛЕЖНО.
Якщо ХОЧА Б ОДНА частина абзацу містить явний конкретний економічний зміст — відповідь YES.

Слова на кшталт "схеми", "корупція", "допомога", "лікарі", "служби" самі по собі НЕ достатні для YES.
YES лише тоді, коли абзац прямо говорить про економічну чи соціальну проблему, економічний ресурс, виплати, ціни, бюджет, фінансування або економічну політику всередині України."""

SYSTEM_PROMPT = """
Ти дослідник публічних промов. Твоє завдання — виділити структуровані
економічні фрейми з абзаців промов (президент, урядовці, ЗМІ і т.п.).

ФРЕЙМ — це одна конкретна соціально-економічна проблема або дія, яку згадують в абзаці.
Один абзац може містити КІЛЬКА фреймів, якщо стосується різних тем, але не більше ЧОТИРЬОХ.
Якщо абзац не містить економічних фреймів — поверни порожній список "frames": [].

ВАЖЛИВО — ПРАВИЛО ОДНОГО ФРЕЙМУ:
Не дроби один загальний меседж на підфрейми!
Якщо абзац описує ОДНУ програму/ініціативу/рішення і перераховує на що вона
поширюється (ліки, комуналка, транспорт тощо) — це ОДИН фрейм про цю програму,
а НЕ окремі фрейми для кожного пункту переліку.
Окремий фрейм виправданий ЛИШЕ якщо:
- інший cause_actor
- інші beneficiaries
- принципово інша проблема (не просто інший товар/послуга в рамках одного рішення)

Для кожного фрейму заповни поля:

- object: конкретний предмет або сфера — тільки іменник, 1-2 слова.
- problem: у чому полягає проблема з цим об'єктом — 1-3 слова.
- short_description: ОДНЕ коротке речення українською за шаблоном: "Йдеться про [сферу/програму] для [кого], у контексті [короткий контекст]".
- cause_actor: хто/що СПРИЧИНЯЄ проблему (або null якщо не вказано)
- government_actions: список конкретних дій/доручень уряду (може бути порожнім [])
- action_status: ОДНЕ з: planned | implemented
- responsible_institutions: список установ — ТІЛЬКИ офіційні абревіатури (КМУ, МОЗ, МОН, Мінфін тощо). Якщо не вказані — [].
- beneficiaries: список з ЛИШЕ двох можливих категорій: "люди/громадяни" та/або "бізнес". Якщо не вказано — [].
- urgency: ОДНЕ з: low/medium | high
- is_economic_frame: true якщо це справді внутрішньоекономічний фрейм, false якщо ні

Відповідай ТІЛЬКИ валідним JSON у форматі: {"frames": [...]}
""".strip()

NARRATIVE_TAGS = [
    # Охорона здоров'я
    'ціни на ліки', 'нестача / доступність ліків',
    'державні закупівлі та реімбурсація ліків',
    'доступна медицина і лікування',
    # Пенсії та соціальний захист
    'індексація пенсій', 'пенсійна реформа',
    'соціальна допомога малозабезпеченим', 'програма єПідтримка / єДопомога дітям', 'субсидії на ЖКГ',
    # Ветерани та оборона
    'реабілітація та протезування ветеранів', 'робочі місця для ветеранів',
    'зарплати військових', 'відновлення деокупованих територій',
    # Освіта
    'шкільне харчування', 'харчування силових структур', 'зарплати вчителів і медиків', 'укриття у школах', 'будівництво та реформа освіти',
    # Енергетика та ЖКГ
    'відновлення електропостачання', 'нові енергетичні потужності',
    'накопичення та закупівля газу', 'опалювальний сезон / теплопостачання', 'тарифи на газ і тепло',
    'тарифи на електроенергію', 'тарифи на воду', 'ціни на бензин / паливо', 'дефіцит / доступність пального', 'нафтогазова промисловість',
    'атомна енергетика', 'генератори та автономне енергозабезпечення', 'водопостачання та водоінфраструктура',
    # Продовольство
    'ціни на продукти харчування', 'продовольча безпека', 'зерновий експорт',
    # Праця та зайнятість
    'зайнятість / безробіття',
    # Бізнес та економіка
    'підтримка малого бізнесу / дерегуляція', 'іноземні інвестиції',
    'переробна промисловість/Зроблено в Україні', 'ВВП та економічне зростання',
    # Фінанси
    'дефіцит держбюджету', 'бюджет регіонів та громад',
    'кредитування бізнесу та населення', 'єОселя / іпотека',
    # Інфраструктура
    'відновлення доріг та мостів', 'Укрзалізниця та транспорт', 'будівництво житла',
    'морська логістика / порти',
    # Цифрові програми та міграція
    'цифровізація держпослуг / Дія', 'єВідновлення / компенсації', 'повернення українців додому',
    # Наука, інклюзія та креативні індустрії
    'наука та інновації', "безбар'єрність", 'культурні індустрії та креативна економіка',
    # Санкції, активи та безпека
    'санкції проти РФ і конфіскація активів', 'захист критичної інфраструктури',
    # Оборонна промисловість
    'оборонна промисловість та закупівлі',
    # Гуманітарна допомога та цивільна безпека
    'пункти незламності та обігріву', 'гуманітарне розмінування',
    # Catch-all
    'інше',
]

TAGS_SYSTEM_PROMPT = (
    'Ти — класифікатор наративів. '
    "Тобі дають об'єкт (object), проблему (problem) і короткий опис контексту (short_description) одного фрейму. "
    'Обери ОДИН найближчий тег із наданого списку. '
    'Використовуй short_description як головне поле для розрізнення схожих object/problem у різних контекстах. '
    'Відповідай тільки назвою тегу, без пояснень.'
)


# ── Хелпери (скопійовано з pipeline.ipynb) ───────────────────────────────────
def para_key(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def frame_tag_key(object_norm: str, problem: str, short_description: str) -> str:
    return hashlib.md5(f'{object_norm}|{problem}|{short_description}'.encode('utf-8')).hexdigest()


def clean_text(value) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return ' '.join(str(v).strip() for v in value.values() if str(v).strip())
    return str(value).strip()


def normalize_str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, list):
        value = [value]
    normalized = []
    for item in value:
        if isinstance(item, dict):
            extracted = ''
            for key in ('action', 'name', 'institution', 'beneficiary', 'value', 'text'):
                extracted = clean_text(item.get(key))
                if extracted:
                    break
            if not extracted:
                extracted = clean_text(' '.join(str(v) for v in item.values()))
            if extracted:
                normalized.append(extracted)
        else:
            text = clean_text(item)
            if text:
                normalized.append(text)
    return normalized


def serialize_list_field(value) -> str:
    return '; '.join(normalize_str_list(value))


# ── Крок 1: розбивка на абзаци ───────────────────────────────────────────────
def split_paragraphs(text: str) -> list:
    paragraphs = []
    for para in text.split('\n'):
        para = para.strip()
        if len(para) >= MIN_PARA_LEN:
            paragraphs.append({
                'paragraph': para,
                'para_id': para_key(para),
            })
    return paragraphs


# ── Крок 3: LLM binary filter (async) ────────────────────────────────────────
async def llm_classify_batch_async(paragraphs_text: list, client: AsyncOpenAI) -> list:
    numbered = '\n\n'.join(f'{i+1}. {p}' for i, p in enumerate(paragraphs_text))
    resp = await client.responses.create(
        model=LLM_MODEL,
        input=[
            {'role': 'system', 'content': LLM_FILTER_PROMPT},
            {'role': 'user', 'content': (
                'Класифікуй кожен абзац. Поверни JSON: {"results": ["YES", "NO", ...]}\n\n'
                + numbered
            )},
        ],
        text={'format': {'type': 'json_object'}},
    )
    parsed = json.loads(resp.output_text)
    for v in parsed.values():
        if isinstance(v, list):
            return [str(x).strip().upper().startswith('Y') for x in v]
    return [False] * len(paragraphs_text)


async def llm_filter(paragraphs: list, client: AsyncOpenAI) -> list:
    results = []
    for i in range(0, len(paragraphs), BATCH_SIZE):
        batch = paragraphs[i:i + BATCH_SIZE]
        texts = [p['paragraph'] for p in batch]
        try:
            flags = await llm_classify_batch_async(texts, client)
        except Exception as e:
            raise RuntimeError(f'LLM filter помилка: {e}') from e
        for para, flag in zip(batch, flags):
            if flag:
                results.append(para)
    return results


# ── Крок 4: Frame extraction (async) ─────────────────────────────────────────
async def extract_frames_async(sem: asyncio.Semaphore, client: AsyncOpenAI, paragraph_text: str, paragraph_id: str) -> list:
    async with sem:
        for attempt in range(3):
            try:
                response = await client.responses.create(
                    model=LLM_MODEL,
                    input=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': f'Абзац (ID: {paragraph_id}):\n{paragraph_text}'},
                    ],
                    text={'format': {'type': 'json_object'}},
                    timeout=120,
                )
                frames = json.loads(response.output_text).get('frames', [])
                if not isinstance(frames, list):
                    return []
                validated = []
                for f in frames:
                    if not isinstance(f, dict):
                        continue
                    f['object'] = clean_text(f.get('object', ''))
                    f['problem'] = clean_text(f.get('problem', ''))
                    f['short_description'] = clean_text(f.get('short_description', ''))
                    f['cause_actor'] = clean_text(f.get('cause_actor')) or None
                    f['government_actions'] = normalize_str_list(f.get('government_actions', []))
                    f['responsible_institutions'] = normalize_str_list(f.get('responsible_institutions', []))
                    f.setdefault('is_economic_frame', True)
                    if f.get('action_status') not in ACTION_STATUSES:
                        f['action_status'] = 'planned'
                    if f.get('urgency') not in URGENCY_LEVELS:
                        f['urgency'] = 'low/medium'
                    raw_ben = normalize_str_list(f.get('beneficiaries', []))
                    f['beneficiaries'] = [b for b in raw_ben if b in BENEFICIARY_CATEGORIES]
                    validated.append(f)
                return validated
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return []
    return []


async def run_step4(paragraphs: list, client: AsyncOpenAI) -> dict:
    sem = asyncio.Semaphore(FRAMES_SEMAPHORE)
    tasks = [extract_frames_async(sem, client, p['paragraph'], p['para_id']) for p in paragraphs]
    results = await asyncio.gather(*tasks)
    return {p['para_id']: frames for p, frames in zip(paragraphs, results)}


# ── Крок 5: Tag assignment (async) ────────────────────────────────────────────
async def assign_tag_async(sem: asyncio.Semaphore, client: AsyncOpenAI, object_norm: str, problem: str, short_description: str) -> str:
    tags_str = '\n'.join(f'- {t}' for t in NARRATIVE_TAGS)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.responses.create(
                    model=LLM_MODEL,
                    input=[
                        {'role': 'system', 'content': TAGS_SYSTEM_PROMPT},
                        {'role': 'user', 'content': f'object: {object_norm}\nproblem: {problem}\nshort_description: {short_description}\n\nСписок тегів:\n{tags_str}'},
                    ],
                    timeout=60,
                )
                tag = resp.output_text.strip().strip('-').strip()
                return tag if tag in NARRATIVE_TAGS else 'інше'
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return 'інше'
    return 'інше'


async def run_step5(frames_dict: dict, client: AsyncOpenAI) -> dict:
    unique_contexts = set()
    for frames in frames_dict.values():
        for frame in frames:
            if frame.get('is_economic_frame', True):
                o = frame.get('object', '').strip().lower()
                p = frame.get('problem', '')
                d = clean_text(frame.get('short_description', ''))
                unique_contexts.add((o, p, d))

    sem = asyncio.Semaphore(TAGS_SEMAPHORE)
    tasks = {(o, p, d): assign_tag_async(sem, client, o, p, d) for o, p, d in unique_contexts}
    results = await asyncio.gather(*tasks.values())
    return {frame_tag_key(o, p, d): tag for (o, p, d), tag in zip(tasks.keys(), results)}


# ── Збірка DataFrame ──────────────────────────────────────────────────────────
def build_dataframe(paragraphs: list, frames_dict: dict, tags_dict: dict) -> pd.DataFrame:
    para_lookup = {p['para_id']: p for p in paragraphs}
    rows = []
    for pid, frames in frames_dict.items():
        if pid not in para_lookup:
            continue
        for fn, frame in enumerate(frames):
            if not frame.get('is_economic_frame', True):
                continue
            obj_text = clean_text(frame.get('object', ''))
            obj_norm = obj_text.lower()
            prob = clean_text(frame.get('problem', ''))
            short_desc = clean_text(frame.get('short_description', ''))
            rows.append({
                'paragraph': para_lookup[pid]['paragraph'],
                'object': obj_text,
                'problem': prob,
                'short_description': short_desc,
                'government_actions': serialize_list_field(frame.get('government_actions', [])),
                'action_status': frame.get('action_status', ''),
                'responsible_institutions': serialize_list_field(frame.get('responsible_institutions', [])),
                'urgency': frame.get('urgency', ''),
                'narrative_tag': tags_dict.get(frame_tag_key(obj_norm, prob, short_desc), 'інше'),
            })
    return pd.DataFrame(rows)


# ── Головна функція пайплайну ─────────────────────────────────────────────────
async def run_pipeline(text: str, api_key: str, progress_cb=None) -> pd.DataFrame:
    client = AsyncOpenAI(api_key=api_key)

    def log(msg):
        if progress_cb:
            progress_cb(msg)

    log('Крок 1: розбивка на абзаци...')
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        raise ValueError('Не знайдено абзаців довше 80 символів.')

    log(f'Крок 3: LLM filter ({len(paragraphs)} абзаців)...')
    econ_paragraphs = await llm_filter(paragraphs, client)
    if not econ_paragraphs:
        return pd.DataFrame()

    log(f'Крок 4: витягування фреймів ({len(econ_paragraphs)} економічних абзаців)...')
    frames_dict = await run_step4(econ_paragraphs, client)

    log('Крок 5: призначення тегів...')
    tags_dict = await run_step5(frames_dict, client)

    log('Готово!')
    return build_dataframe(econ_paragraphs, frames_dict, tags_dict)
