"""Hard-coded Bangla test sentences for the noise-suite tests and the Table I / CER report.

Kept here (not downloaded) so the tests never depend on data access.

# TODO(verify): NATIVE-SPEAKER CHECK NEEDED.
#   These sentences were written to exercise the machinery, not vetted line by
#   line for natural phrasing. What matters for the CODE is only that each is
#   valid Bangla Unicode containing the relevant features (conjuncts, matras,
#   nukta letters ড়/ঢ়/য়, homophone-prone letters, digits, emoji). Please:
#     1. read each line and fix anything that is not natural Bangla,
#     2. confirm SENTENCES_NUKTA really do contain ড় / ঢ় / য়,
#     3. confirm SENTENCES_EMOJI render the emoji you expect.
#   Replacing any sentence with a real corpus sentence is fine and encouraged.
"""

from __future__ import annotations

# --- general sentences: conjuncts, matras, homophone-prone letters ----------
SENTENCES_GENERAL: list[str] = [
    "আমি প্রতিদিন সকালে বিদ্যালয়ে যাই।",
    "বাংলাদেশের রাজধানী ঢাকা একটি ব্যস্ত শহর।",
    "সে খুব ভালো ছাত্র এবং পরিশ্রমী।",
    "আজকে আকাশে অনেক মেঘ জমেছে।",
    "তুমি কি আগামীকাল আমাদের বাড়িতে আসবে?",
    "এই বইটি আমার খুব পছন্দ হয়েছে।",
    "রাষ্ট্রের উন্নয়নে শিক্ষার ভূমিকা অপরিসীম।",
    "গ্রামের মানুষ কৃষিকাজের উপর নির্ভরশীল।",
    "বিজ্ঞান ও প্রযুক্তির অগ্রগতি আমাদের জীবন সহজ করেছে।",
    "সে গতকাল ট্রেনে করে চট্টগ্রাম গিয়েছিল।",
    "আমার ছোট ভাই ফুটবল খেলতে ভালোবাসে।",
    "নদীর পানি এখন অনেক বেড়ে গেছে।",
    "দুর্নীতি একটি জাতির অগ্রগতির প্রধান অন্তরায়।",
    "চিকিৎসকেরা রোগীদের সেবায় নিয়োজিত থাকেন।",
    "বসন্তকালে গাছে গাছে নতুন পাতা গজায়।",
    "পরীক্ষার ফলাফল আগামী সপ্তাহে প্রকাশিত হবে।",
    "আমরা সবাই মিলে অনুষ্ঠানটি সফল করেছি।",
    "তার কথা শুনে সবাই অবাক হয়ে গেল।",
    "শীতকালে খেজুরের রস খেতে খুব মজা।",
    "এই রাস্তাটি সংস্কারের কাজ চলছে।",
    "মা রান্নাঘরে ভাত আর মাছ রান্না করছেন।",
    "ছেলেটি বড় হয়ে একজন প্রকৌশলী হতে চায়।",
    "সরকার নতুন একটি প্রকল্প ঘোষণা করেছে।",
    "বইমেলায় প্রতিদিন প্রচুর মানুষ ভিড় করে।",
    "পাখিরা ভোরবেলা মধুর সুরে গান গায়।",
    "তথ্যপ্রযুক্তি খাতে বাংলাদেশ দ্রুত এগিয়ে যাচ্ছে।",
    "আমার দাদা প্রতিদিন সকালে হাঁটতে বের হন।",
    "খেলাটি বৃষ্টির কারণে বন্ধ রাখতে হয়েছে।",
    "শিক্ষকেরা শিক্ষার্থীদের নৈতিক শিক্ষা দেন।",
    "এই অঞ্চলের মাটি ধান চাষের জন্য উপযুক্ত।",
    "অনেক দিন পর পুরোনো বন্ধুর সাথে দেখা হলো।",
    "নতুন প্রজন্ম প্রযুক্তির সাথে দ্রুত মানিয়ে নেয়।",
    "সন্ধ্যায় বাজারে গিয়ে কিছু শাকসবজি কিনলাম।",
    "তিনি একজন সৎ ও নিষ্ঠাবান কর্মকর্তা।",
    "রোদে পুড়ে কৃষক মাঠে কাজ করে চলেছেন।",
    "আমরা ছুটিতে সমুদ্রসৈকতে বেড়াতে গিয়েছিলাম।",
    "হঠাৎ বিদ্যুৎ চলে যাওয়ায় পুরো এলাকা অন্ধকার হয়ে গেল।",
    "শহরের যানজট দিন দিন বেড়েই চলেছে।",
    "মানুষের প্রতি সহানুভূতিই প্রকৃত মনুষ্যত্ব।",
    "চাঁদের আলোয় পুরো মাঠ ঝলমল করছিল।",
]

# --- nukta letters ড় (U+09DC) / ঢ় (U+09DD) / য় (U+09DF) -------------------
# These decompose under NFC to <base, U+09BC>, i.e. two code points in one
# grapheme cluster. A code-point-level slip in any noise type shows up here first.
SENTENCES_NUKTA: list[str] = [
    "বড় গাড়িতে চড়ে সে দূরের পথ পাড়ি দিল।",
    "আষাঢ় মাসের বৃষ্টিতে মাঠঘাট কাদায় ভরে গেছে।",
    "দৃঢ় মনোবল থাকলে যেকোনো বাধা পেরোনো যায়।",
    "নয় জন শিক্ষার্থী বিষয়টি সময়মতো শেষ করেছে।",
    "পড়াশোনায় মন দিলে ভালো ফল পাওয়া যায়।",
    "ছেলেটি গাছ থেকে পড়ে গিয়ে হাত ভেঙে ফেলেছে।",
    "পাহাড়ি এলাকায় বড় বড় পাথর গড়িয়ে পড়ছিল।",
    "সময় বড় নিষ্ঠুর, কারও জন্য অপেক্ষা করে না।",
]

# --- emoji: N10 conditions -------------------------------------------------
SENTENCES_EMOJI: list[str] = [
    "আজকের ম্যাচটা দারুণ ছিল 🔥🔥 দল জিতেছে 🎉",
    "এত সুন্দর খবর শুনে মন ভরে গেল 😊❤️",
    "পরীক্ষা শেষ 😭 এখন শুধু বিশ্রাম করব 😴",
    "নতুন চাকরিটা পেয়ে খুব খুশি লাগছে 🙏✨",
    "এই সিদ্ধান্তটা একদম ঠিক হয়নি 😡👎",
]

# --- embedded English, digits, mentions, URLs (realistic social-media noise) -
SENTENCES_MIXED: list[str] = [
    "COVID-19 মহামারির সময় অনলাইন ক্লাস জনপ্রিয় হয়েছিল।",
    "আগামী ২০৩০ সালের মধ্যে দেশ ডিজিটাল হবে বলে আশা করা হচ্ছে।",
    "@user আপনার পোস্টটি খুবই তথ্যবহুল ছিল, ধন্যবাদ।",
    "বিস্তারিত জানতে https://example.com ওয়েবসাইটে যান।",
    "গত ম্যাচে সে ৭৫ রান করে ম্যান অব দ্য ম্যাচ হয়েছে।",
]

#: Everything, in a stable order.
SENTENCES: list[str] = (
    SENTENCES_GENERAL + SENTENCES_NUKTA + SENTENCES_EMOJI + SENTENCES_MIXED
)
