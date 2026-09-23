"""FAQ direct-answer shortlist — bypass retrieval for curated questions.

When a user query closely matches one of these FAQs, the RAG pipeline
skips KB retrieval and answers directly from the curated content via LLM.
This keeps official/support answers deterministic and avoids retrieval noise.

Matching is intentionally fuzzy: exact → substring → token containment →
difflib → embedding cosine. The user should not need character-perfect phrasing.
"""

from __future__ import annotations

import logging
import difflib

log = logging.getLogger(__name__)

# Curated Q → A pairs (exactly as provided by product). Keys are the
# canonical question phrasing; values are the authoritative answers.
FAQ_SHORTLIST: dict[str, str] = {
    "اگر فکر می کنم اطلاعات گزارش اعتباری من اشتباه است، چه کاری باید انجام بدهم؟": "لطفا جهت راهنمایی برای اصلاح اطلاعات گزارش اعتبارسنجی خود به سامانه پشتیبانی شرکت اعتبارسنجی ایران مراجعه فرمایید:\nwww.Support.ics24.ir",
    "مدیرعامل شرکت اعتبارسنجی ایران کیست؟": "دکتر رضا قاسم‌پور در حال حاضر مدیرعامل شرکت اعتبارسنجی ایران است. این شخص دانش‌آموخته دکتری ریاضی کاربردی و کارشناسی ارشد اقتصاد از دانشگاه تهران بوده و پیش از این مدیر طرح و برنامه بانک رفاه کارگران و عضو هیئت مدیره شرکت اعتبارسنجی ایران بوده است.\nدر دوره مدیریت این شخص، نسخه جدید مدل امتیازدهی اعتباری با پوشش ۱۰۰ درصدی ارائه گردید و اعتبارسنجی به حوزه چک نیز وارد شد.",
    "اطلاعات موجود در گزارش اعتباری هر چند وقت یک‌بار به روزرسانی می‌شوند؟": "اطلاعات مربوط به وام‌ها و تسهیلات بانکی به‌صورت ماهانه از سوی بانک‌ها ارسال و به روز می‌شود؛ اما اطلاعات چک‌های برگشتی، بدهی‌های مالیاتی و احکام قضایی ظرف ۲۴ ساعت پس از تغییر یا رفع سوءاثر به روزرسانی می‌گردند.",
    "شرکت اعتبارسنجی ایران را معرفی کن.": "شرکت اعتبارسنجی ایران (سهامی خاص) در آبان‌ماه ۱۳۸۵ بر اساس ماده (۵) قانون تسهیل اعطای تسهیلات بانکی و آیین‌نامه نظام سنجش اعتبار تأسیس شده است.\nارزش‌های محوری شرکت شامل «بی‌طرفی»، «رعایت محرمانگی اطلاعات» و «دقت و درستی» است. مأموریت اصلی شرکت نیز ایجاد اعتماد در بازارها، اقتصاد و جامعه از طریق تحلیل هوشمند رفتار مالی و تصمیم‌گیری آگاهانه تعریف شده است.\nشرکت در سال ۱۳۹۵ اولین مدل امتیاز اعتباری را با همکاری شرکت CreditInfo ارائه کرد. پس از بومی‌سازی سامانه در سال ۱۳۹۸، در سال ۱۴۰۳ نسخه دوم گزارش و امتیاز اعتباری مبتنی بر هوش مصنوعی توسط شرکت راه اندازی شد و در سال ۱۴۰۴ نیز گزارش و امتیاز اعتبارسنجی چک به بهره‌برداری رسید.\nدر ادامه، شرکت اعتبارسنجی ایران در حال پیاده سازی محصولات مختلفی از جمله نسخه جدید مدل امتیاز اعتباری افراد و کسب و کارها، حد اعتباری تسهیلات و سامانه اعتبارسنجی معاملات می باشد.",
    "اگر چک برگشتی من رفع سوء اثر شده باشد، اما همچنان در گزارش اعتباری نمایش داده شود، چه اقداماتی باید انجام دهم تا از گزارش اعتباری من حذف شود؟": "حداکثر تا ۲۴ ساعت بعد از رفع سوءاثر چک، گزارش اعتباری شما بروزرسانی شده و شما می توانید گزارش جدید خود را دریافت کنید. البته لازم به ذکر است که حتی پس از رفع سوءاثر چک برگشتی، سوابق آن در گزارش اعتباری درج خواهد شد و در امتیاز شما تاثیرگذار است.",
    "اگر هیچ وامی نگرفتم رتبه اعتباریم چند حساب میشه؟": "در مدل امتیاز اعتباری، در صورتی که هیچ تسهیلاتی دریافت نکرده باشید، بر اساس سایر اطلاعات تکمیلی نظیر مالیات، صدک درآمدی خانوار، چکهای برگشتی و نقدشده، امتیاز اعتباری شخص محاسبه میشود. در حال حاضر، در صورت نداشتن وضعیت منفی نظیر محکومیت مالی اجرانشده با مبلغ بالا، چک برگشتی سوءاثرنشده و بدهی مالیاتی و همچنین نداشتن سابقه اخذ تسهیلات در شبکه بانکی، رتبه اعتباری شما در بازه C1 تا C3 خواهد بود. برای دریافت گزارش اعتباری خود به آدرس اینترنتی https://etebarito.ics24.ir مراجعه نمایید.",
    "هدف اصلی از اعتبارسنجی چیست؟": "ارزیابی میزان ریسک افراد و کسب و کارها بر اساس سوابق مالی گذشته، جهت برقراری عدالت در ارائه خدمات مالی و هدایت جامعه به سمت انضباط مالی است.",
    "راه‌های رسمی دریافت گزارش و ارتباط با واحد پشتیبانی چیست؟": "دریافت گزارش از طریق سامانه وب‌سایت اعتباریتو (etebarito.ics24.ir) صورت می‌گیرد و پشتیبانی مشتریان از طریق شماره تلفن ۰۲۱۴۱۵۲۲۲۲۲ یا سامانه ثبت تیکت support.ics24.ir پاسخگوی کاربران است.",
    "تفاوت اثر چک برگشتی «رفع‌سوءاثرنشده» با «رفع‌سوءاثرشده» در امتیاز چک چیست؟": "چک برگشتی رفع‌سوءاثرنشده جزئی از خطوط قرمز اعتباری است و امتیاز چک صادرکننده را مستقیماً به حداقل ممکن یعنی عدد ۱ (رتبه E3) می‌رساند. اما پس از رفع سوءاثر، فرد از این وضعیت خارج شده و اثر منفی سابقه آن ظرف ۵ سال به تدریج کاهش می‌یابد.",
    "چه عواملی امتیاز اعتباری من را کاهش می‌دهد؟": "به‌طور خلاصه، هر رفتاری که از دید نظام مالی کشور نشان‌دهنده‌ی افزایش ریسک عدم ایفای تعهدات باشد، باعث کاهش امتیاز اعتباری خواهد شد. در نتیجه:\nهرگونه تأخیر در پرداخت اقساط تسهیلات، حتی کوتاه‌مدت، می‌تواند امتیاز را کاهش دهد. همچنین وجود سابقه چک برگشتی و تاخیر در رفع سوءاثر آن، امتیاز شما را کاهش می دهد.\nقرار گرفتن در وضعیت‌های منفی تسهیلات مانند سررسیدگذشته، معوق، مشکوک‌الوصول یا سوخت‌شده نشان‌دهنده افزایش ریسک اعتباری شخص بوده و باعث کاهش جدی امتیاز و رتبه اعتباری می‌شود.\nورود پرونده‌های مالیاتی به مرحله‌ی وصول و اجرا، داشتن محکومیت مالی یا ورشکستگی، و تعهد ارزی رفع‌نشده نیز از دیگر عوامل مستقیم افت امتیاز و رتبه هستند.\nلازم به ذکر است که تاثیر هر کدام از رفتارهای منفی ذکر شده در امتیاز اعتباری با تحلیل داده و به کارگیری هوش مصنوعی به دست می آید.",
    "آیا داشتن رتبه اعتباری بالا (مانند A1) دریافت وام از بانک را تضمین می‌کند؟": "گزارش اعتبارسنجی ابزاری مشورتی برای ارزیابی ریسک است و تصمیم‌گیری نهایی درباره اعطای وام و شرایط آن کاملاً بر عهده سیاست‌های داخلی خود بانک است.",
    "امتیاز اعتباری چیست و با رتبه اعتباری چه تفاوتی دارد؟": "امتیاز اعتباری یک عدد بین ۲۵۰ تا ۹۰۰ است که نشان می دهد شخص یا کسب و کار چقدر در بازپرداخت تعهدات خود خوش حساب است و ریسک اعتباری اش چقدر است. رتبه، در واقع دسته‌بندی همین امتیاز است (از A1 به عنوان بهترین تا E3 به عنوان ضعیف‌ترین.)",
    "سابقه تسهیلات و وام چند سال در گزارش اعتباری باقی می ماند؟": "سوابق تسهیلاتی، چک و مالیاتی شرکت‌ها تا ۵ سال در پایگاه داده محاسبه می‌شود. اما با گذشت زمان اثر مثبت و منفی آن‌ کمرنگ‌تر شده و پس از ۵ سال کلاً حذف می‌گردند.",
    "چه مواردی باعث می شود امتیاز اعتباری شرکت ۲۵۰ شود؟": "هر یک از موارد زیر در زمان استعلام، امتیاز شرکت را به ۲۵۰ (حداقل امتیاز ممکن) می‌رساند:\n1-چک برگشتی رفع سوءاثر نشده\n2-بدهی مالیاتی\n3-محکومیت مالی\n4-ورشکستگی و اعسار\n5-اقساط معوق و مشکوک‌الوصول",
    "چه مواردی باعث می شود امتیاز اعتباری من ۲۵۰ شود؟": "هر یک از موارد زیر در زمان استعلام، امتیاز شما را به ۲۵۰ و رتبه‌تان را به E3 می‌رساند:\n1-چک برگشتی رفع سوءاثر نشده\n2-بدهی مالیاتی (برای دریافت جزئیات بیشتر درباره پرونده های مالیاتی خود، به وبسایت my.tax.gov مراجعه کنید)\n3-محکومیت مالی اجرانشده\n4-ورشکستگی و اعسار\n5-اقساط معوق و مشکوک‌الوصول\n6-تاخیر در بازپرداخت بیش از ۹۰ روز متوالی\nلطفا توجه داشته باشید که ضامن یک شخص بد حساب بودن به تنهایی نمی تواند باعث رتبه E3 شود و تنها امتیاز شما را مقداری کاهش می دهد.",
    "خطوط قرمز امتیاز اعتباری چیست؟": "خطوط قرمز رفتارهایی هستند که باعث می شود شخص کمترین امتیاز ممکن را دریافت کند.",
    "خطوط قرمز امتیاز اعتباری برای افراد چیست؟": "هر یک از موارد زیر در زمان استعلام، امتیاز شما را به ۲۵۰ و رتبه‌تان را به E3 می‌رساند:\n1-چک برگشتی رفع سوءاثر نشده\n2-بدهی مالیاتی (برای دریافت جزئیات بیشتر درباره پرونده های مالیاتی خود، به وبسایت my.tax.gov مراجعه کنید)\n3-محکومیت مالی اجرانشده\n4-ورشکستگی و اعسار\n5-اقساط معوق و مشکوک‌الوصول\n6-تاخیر در بازپرداخت بیش از ۹۰ روز متوالی\nلطفا توجه داشته باشید که ضامن یک شخص بد حساب بودن به تنهایی نمی تواند باعث رتبه E3 شود و تنها امتیاز شما را مقداری کاهش می دهد.",
    "خطوط قرمز امتیاز اعتباری برای کسب و کارها چیست؟": "هر یک از موارد زیر در زمان استعلام، امتیاز شرکت را به ۲۵۰ (حداقل امتیاز ممکن) می‌رساند:\n1-چک برگشتی رفع سوءاثر نشده\n2-بدهی مالیاتی\n3-محکومیت مالی\n4-ورشکستگی و اعسار\n5-اقساط معوق و مشکوک‌الوصول",
    "جدیدا وام گرفته ام. امتیازم چطور می شود؟": "در این حالت، از داده‌های جایگزین مثل صدک درآمدی، سوابق مالیاتی و وضعیت ضمانت شما جهت محاسبه امتیاز اعتباری استفاده می شود و در این حالت، رتبه شما بین C1 تا C3 است.\nپس از ۵ ماه پرداخت قسط، امتیاز شما بر اساس مدل اصلی محاسبه خواهد شد و در صورت بازپرداخت به موقع اقساط، رتبه شما از C فراتر می رود.",
    "به عدد امتیازم اعتراض دارم. چه اقدامی باید انجام دهم؟": "امتیاز و رتبه اعتباری، حاصل محاسبات آماری و مدل‌های هوش مصنوعی و بر اساس سوابق مالی شماست. به همین دلیل، امکان اعتراض مستقیم به عدد امتیاز یا تغییر آن توسط کارشناسان ما وجود ندارد. امتیاز شما تنها در صورتی تغییر می‌کند که اطلاعات درج‌شده در گزارش شما به روز و یا اصلاح شوند.\nهمچنین، در گزارش اعتباری شما، بخشی تحت عنوان «دلایل کاهش امتیاز» وجود دارد که ۵ دلیل اصلی کاهش امتیاز شما را نشان می دهد.",
    "آیا با امتیاز اعتباری پایین می توانم وام بگیرم؟": "بانک‌ها و دیگر وام دهندگان طبق قانون ملزم به استفاده از ابزار اعتبارسنجی هستند، اما تصمیم نهایی برای اعطای وام با خود بانک است. بسته به سیاست‌های داخلی هر بانک، ممکن است با وجود رتبه پایین، با دریافت ضمانت و وثیقه بیشتر به شما تسهیلات اعطا شود.",
    "چگونه می توانم امتیازم را افزایش دهم؟": "اگر در حال حاضر وام در جریان دارید: کلید اصلی افزایش امتیاز، تداوم رفتار اعتباری مثبت است. پرداخت منظم و به‌موقع اقساط، تسویه بدهی‌های سررسید شده پرداخت نشده، رفع سوءاثر چک‌ها، جلوگیری از برگشت چک، پرداخت بدهی مالیاتی و پرداخت مبلغ محکومیت مالی، مهم‌ترین عوامل افزایش امتیاز و رتبه اعتباری هستند.\nاگر در حال حاضر وام در جریان ندارید:\nدریافت یک وام و پرداخت منظم آن حداقل به مدت ۴ ماه، شما را به رتبه‌های بالاتر از C می‌رساند.",
    "اطلاعات گزارش اعتباری من در چه زمانی به روزرسانی می شود؟": "به جز اطلاعات تسهیلاتی که به صورت ماهانه به روزرسانی می شود، دیگر اطلاعات (مالیات، چک و محکومیت های مالی)، پس از ۲۴ ساعت به روز رسانی می شود.\nبرای مشاهده تاریخ به روزرسانی اطلاعات تسهیلات بانک ها و سایر تامین کنندگان، به لینک زیر مراجعه کنید:\nhttps://ics24.ir/about/updatecreditor",
    "هدف از اعتبارسنجی چیست؟": "در گذشته، اعتبار و شهرت افراد در جامعه به واسطه آشنایی‌های شخصی و تجربیات فردی در مراوده با یکدیگر شکل می‌گرفت. عواملی نظیر رشد جمعیت، بزرگ‌تر شدن جوامع، پیچیده‌تر شدن روابط انسانی و افزایش تنوع در معاملات باعث شدند تا افراد شناخت کافی درباره‌ی کسانی که با آن‌ها در تعامل هستند، نداشته باشند.\nدر این راستا، شرکت‌های اعتبارسنجی در دنیا برای رفع نیاز به سنجش اعتبار افراد تاسیس شده و این کار را با جمع‌آوری، پردازش و تحلیل اطلاعات اعتباری افراد از منابع اطلاعاتی مختلف انجام می دهند.",
    "چگونه گزارش خلاصه چک صادر کننده چک را دریافت کنم؟": "اگر شما گیرنده چک باشید، برای دریافت گزارش اعتبارسنجی شخص صادر کننده چک (تا قبل از نقد شدن یا انتقال چک) نیازی به اجازه صادرکننده نیست و فقط به شناسه صیادی ۱۶ رقمی نیاز دارید. برای دریافت گزارش چک صادر کننده به وبسایت https://etebarito.ir مراجعه کنید.",
    "اعضای هیئت مدیره شرکت اعتبارسنجی ایران چه اشخاصی هستند؟": "اعضای هیئت مدیره شرکت اعتبارسنجی ایران به شرح زیر است:\nدکتر رضا قاسم پور: مدیرعامل شرکت اعتبارسنجی ایران. این شخص دانش‌آموخته دکتری ریاضی کاربردی و کارشناسی ارشد اقتصاد از دانشگاه تهران بوده و پیش از این مدیر طرح و برنامه بانک رفاه کارگران و عضو هیئت مدیره شرکت اعتبارسنجی ایران بوده است.\nدر دوره مدیریت این شخص، نسخه جدید مدل امتیازدهی اعتباری با پوشش ۱۰۰ درصدی ارائه گردید و اعتبارسنجی به حوزه چک نیز وارد شد.\nدکتر کمیل شاعری: این شخص رئیس هیئت مدیره و دانش آموخته دکتری مدیریت مالی می باشند و پیش از این، معاون سرمایه‌گذاری صندوق نوآوری و شکوفایی و رئیس هیئت‌مدیره صندوق پژوهش و فناوری نوآفرین (بزرگ‌ترین صندوق پژوهش و فناوری کشور) بوده است.\nدکتر محسن صادقی: دانش‌آموخته دکتری مدیریت مالی از دانشگاه شهید بهشتی ایران و نایب رئیس هیئت مدیره. پیش از این، ایشان در سمت مدیریت ریسک و انطباق در بانک سرمایه‌گذاری نوین و به عنوان پژوهشگری مالی در سازمان بورس و اوراق بهادار مشغول به فعالیت بوده اند.\nمهدی احمدی: مهدی احمدی، عضو هیئت مدیره شرکت اعتبارسنجی ایران و مدیرعامل شرکت رایان هم افزا می باشند.",
    "اقساط یکی از وام های من در گزارش اعتباری یافت نمی شود. چه اقدامی انجام دهم؟": "این موضوع را به واحد ارتباط با مشتریان شرکت اعتبارسنجی ایران با شماره ۰۲۱۴۱۵۲۲۲۲۲ اطلاع دهید تا بررسی گردد.",
    "چگونه با شرکت اعتبارسنجی ایران تماس بگیرم؟": "برای تماس با واحد ارتباط با مشتریان شرکت اعتبارسنجی ایران با شماره ۰۲۱۴۱۵۲۲۲۲۲ تماس بگیرید.",
    "کجا گزارش اعتباری و گزارش چک خودم را بگیرم؟": "برای این کار، به وبسایت اعتباریتو مراجعه فرمایید.\nwww.etebarito.ics24.ir",
    "مدیران شرکت اعتبارسنجی ایران چه اشخاصی هستند؟": "برای اطلاع از ساختار معاونتی و مدیریتی شرکت اعتبارسنجی ایران، به لینک زیر مراجعه فرمایید:\nhttps://ics24.ir/about/managerteam",
}

# Normalized lookup — precomputed for fast exact match
def _normalize(text: str) -> str:
    """Persian-aware normalization for FAQ matching (mirrors retrieve.py)."""
    t = (text or "").strip().lower()
    # unify ZWNJ, yeh/kaf variants, and common compounds
    t = t.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    # handle compound سوءاثر variants without space
    t = t.replace("سوءاثر", "سوء اثر").replace("سواثر", "سوء اثر")
    # informal shortcuts & verb variations
    t = t.replace("چیکار", "چه کار").replace("چطور", "چگونه").replace("کی", "چه زمانی")
    t = t.replace("گرفتم", "گرفته").replace("گرفتیم", "گرفته").replace("گرفتید", "گرفته")
    t = t.replace("جدید ", "جدیدا ").replace("جدیداا", "جدیدا")
    t = t.replace("میشه", "می شود").replace("میشود", "می شود").replace("میشم", "می شود")
    t = t.replace("چی میشه", "چه می شود").replace("چی می شود", "چه می شود")
    # strip common punctuation (including Persian ؟ ،)
    for ch in "؟?!.,،؛:؛\"'«»()[]{}…-–—\n\r\t":
        t = t.replace(ch, " ")
    # collapse whitespace
    t = " ".join(t.split())
    return t


# Build normalized → answer + original question map
_NORMALIZED_FAQ: dict[str, dict[str, str]] = {}
for _q, _a in FAQ_SHORTLIST.items():
    _norm = _normalize(_q)
    _NORMALIZED_FAQ[_norm] = {"answer": _a, "question": _q}

# --- IDF for token rarity (to avoid generic "اعتبارسنجی چیست" false positives) ---
import math
from collections import Counter
_N = len(_NORMALIZED_FAQ)
_DF = Counter()
for norm_q in _NORMALIZED_FAQ:
    toks = set(norm_q.split())
    for t in toks:
        if len(t) > 1:
            _DF[t] += 1
_IDF = {t: math.log(_N / (df)) for t, df in _DF.items()}

# --- Optional embedding index for fuzzy semantic matching ---
_EMBED_MODEL = None
_EMBED_DIM = 384
_FAQ_EMBEDDINGS: dict[str, list[float]] | None = None  # norm_q -> vector
_EMBED_THRESHOLD = 0.72  # cosine similarity threshold for FAQ hit

def _try_load_embed_model():
    global _EMBED_MODEL, _FAQ_EMBEDDINGS
    if _EMBED_MODEL is not None or _FAQ_EMBEDDINGS is not None:
        return
    try:
        # offline-prep stores model with _ instead of /
        import os
        from sentence_transformers import SentenceTransformer
        candidates = [
            "/splunk-data/v1/Work_RAG-Server-Setup/offline-prep/models/huggingface/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2",
            "/splunk-data/v1/Work_RAG-Server-Setup/offline-prep/models/huggingface/sentence-transformers_all-MiniLM-L6-v2",
        ]
        model_path = next((p for p in candidates if os.path.isdir(p)), None)
        if model_path is None:
            # fall back to HF cache name (will try hub offline)
            model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        _EMBED_MODEL = SentenceTransformer(model_path, device="cpu", trust_remote_code=False)
        # pre-compute FAQ embeddings (normalized questions)
        questions = list(_NORMALIZED_FAQ.keys())
        vectors = _EMBED_MODEL.encode(questions, normalize_embeddings=True, show_progress_bar=False)
        _FAQ_EMBEDDINGS = {q: v.tolist() if hasattr(v, "tolist") else list(v) for q, v in zip(questions, vectors)}
        log.info("FAQ embed index built: %d entries dim=%d from %s", len(_FAQ_EMBEDDINGS), _EMBED_DIM, model_path)
    except Exception as e:
        log.warning("FAQ embedding index not available (%s) — falling back to token/difflib only", e)
        _EMBED_MODEL = None
        _FAQ_EMBEDDINGS = None


# Lazily try once at import — non-blocking; if it fails we still have token/difflib
try:
    _try_load_embed_model()
except Exception:
    pass


def _cosine(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    # vectors are L2-normalized, so dot is cosine
    return float(dot)


# Also index without trailing question particles for looser match
# e.g. "مدیرعامل شرکت اعتبارسنجی ایران کیست" vs "مدیرعامل ... کیست؟"
def find_direct_answer(query: str) -> str | None:
    """
    Return canned answer if query matches a FAQ entry, else None.

    Matching strategy (in order, fuzzy):
      1. exact normalized match
      2. substring containment (len>8, ratio>=0.5)
      3. token-set containment (>=60% of smaller set, >=3 tokens)
      4. difflib SequenceMatcher ratio >=0.72
      5. embedding cosine >=0.72 (paraphrase-multilingual-MiniLM-L12-v2)
    """
    nq = _normalize(query)
    if not nq:
        return None

    # 1) exact normalized match
    if nq in _NORMALIZED_FAQ:
        return _NORMALIZED_FAQ[nq]["answer"]

    # 2) substring containment — balanced: either string contains the other
    for norm_q, payload in _NORMALIZED_FAQ.items():
        if len(norm_q) > 8 and len(nq) > 8 and (norm_q in nq or nq in norm_q):
            longer = max(len(norm_q), len(nq))
            shorter = min(len(norm_q), len(nq))
            # stricter ratio to avoid generic "اعتبارسنجی چیست" matching longer FAQ
            if shorter / longer >= 0.75 and abs(len(norm_q) - len(nq)) <= 15:
                return payload["answer"]

    # 3) embedding semantic similarity — handles true paraphrases, before token to pick correct FAQ
    # e.g. "اطلاعات گزارش اعتباری کی به روز میشه؟" vs "اطلاعات موجود در گزارش اعتباری هر چند وقت یک‌بار به روزرسانی می‌شوند؟"
    # Guard: very short/generic queries (<20 chars or <3 tokens) skip embedding to avoid false positives
    try:
        if len(nq) >= 20:
            q_tokens_tmp = set(t for t in nq.split() if len(t) > 1)
            if len(q_tokens_tmp) >= 3:
                if _FAQ_EMBEDDINGS is None:
                    _try_load_embed_model()
                if _EMBED_MODEL is not None and _FAQ_EMBEDDINGS is not None:
                    q_vec = _EMBED_MODEL.encode([nq], normalize_embeddings=True, show_progress_bar=False)[0]
                    q_list = q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec)
                    best_score = -1.0
                    best_emb_payload = None
                    for norm_q, payload in _NORMALIZED_FAQ.items():
                        vec = _FAQ_EMBEDDINGS.get(norm_q)
                        if vec is None:
                            continue
                        score = _cosine(q_list, vec)
                        if score > best_score:
                            best_score = score
                            best_emb_payload = payload
                    # higher threshold to avoid generic hits (0.78)
                    if best_score >= 0.85 and best_emb_payload is not None:
                        log.info("FAQ embedding hit score=%.3f for query=%r -> %r", best_score, query[:60], best_emb_payload["question"][:40])
                        return best_emb_payload["answer"]
    except Exception as e:
        log.warning("FAQ embedding check failed: %s", e)

    # 4) token-set containment with IDF weighting — pick best FAQ
    q_tokens = set(t for t in nq.split() if len(t) > 1)
    best_tok_score = 0
    best_tok_payload = None
    for norm_q, payload in _NORMALIZED_FAQ.items():
        faq_tokens = set(t for t in norm_q.split() if len(t) > 1)
        if not faq_tokens or not q_tokens:
            continue
        inter = q_tokens & faq_tokens
        if len(inter) < 2:
            continue
        # guard: generic 2-token common queries
        if len(q_tokens) <= 2 and len(inter) <= 2:
            if all(_DF.get(t, 0) > _N * 0.2 for t in inter):
                continue
        # allow 2-token only if one is rare
        if len(inter) == 2 and not any(_DF.get(t, 0) <= 3 for t in inter):
            if len(q_tokens) <= 3:
                continue
        faq_idf_sum = sum(_IDF.get(t, 1.0) for t in faq_tokens)
        inter_idf_sum = sum(_IDF.get(t, 1.0) for t in inter)
        idf_ratio = inter_idf_sum / faq_idf_sum if faq_idf_sum else 0
        smaller = min(len(q_tokens), len(faq_tokens))
        cnt_ratio = len(inter) / smaller if smaller else 0
        jaccard = len(inter) / len(q_tokens | faq_tokens) if (q_tokens | faq_tokens) else 0
        score = max(idf_ratio*1.2, cnt_ratio, jaccard*1.1)
        # boost rare overlap
        if any(_DF.get(t, 0) <= 2 for t in inter):
            score += 0.15
        if len(faq_tokens) <= 4 and len(inter) >= len(faq_tokens) - 1:
            score = max(score, 0.75)
        if score > best_tok_score:
            best_tok_score = score
            best_tok_payload = payload
    if best_tok_payload is not None and best_tok_score >= 0.60:
        return best_tok_payload["answer"]

    # 5) difflib fuzzy string ratio — stricter to avoid generic hits
    best_ratio = 0.0
    best_payload = None
    for norm_q, payload in _NORMALIZED_FAQ.items():
        r = difflib.SequenceMatcher(None, nq, norm_q).ratio()
        if r > best_ratio:
            best_ratio = r
            best_payload = payload
    if best_ratio >= 0.80 and best_payload is not None:
        if len(nq) >= 15 and len(nq.split()) >= 3:
            return best_payload["answer"]

    return None


def is_faq_query(query: str) -> bool:
    """Convenience: does this query belong to the FAQ shortlist?"""
    return find_direct_answer(query) is not None


def list_faq_questions() -> list[str]:
    """Return canonical FAQ question list (for debug/dashboard)."""
    return list(FAQ_SHORTLIST.keys())
