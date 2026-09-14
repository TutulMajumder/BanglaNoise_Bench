# bangla-noisebench — rendered constants sheet

Rendered from `noisebench/bangla_maps.py` by `scripts/constants_sheet.py`. Every `# TODO(verify):` block in that file still needs a native-speaker pass; this sheet only makes the values readable and checks their structure.

**Structural assertions: ALL PASS**

## Assertions

| check | result | detail |
|-------|--------|--------|
| consonant pool size == 35 | PASS | got 35 |
| independent_vowel pool size == 11 | PASS | got 11 |
| digit pool size == 10 | PASS | got 10 |
| IN_CLASS_POOLS match BANGLA_* inventories | PASS |  |
| no matra / hasanta in any IN_CLASS_POOL | PASS |  |
| no matra / hasanta in INSERTABLE_CHARS | PASS |  |
| no chandrabindu (U+0981) in INSERTABLE_CHARS | PASS |  |
| no digits in INSERTABLE_CHARS (removed 2026-09-01: no plausible mechanism for a digit-in-word insertion) | PASS |  |
| IN_CLASS_POOLS are mutually disjoint | PASS |  |
| every IN_CLASS_POOL has >= 2 members (N3 can always pick a different char) | PASS |  |
| HOMOPHONE_SETS members are distinct across sets | PASS |  |
| every HOMOPHONE set has >= 2 members | PASS |  |
| ি/ী and ু/ূ are present as HOMOPHONE sets (M0 Q3) | PASS |  |
| ই/ঈ, উ/ঊ, র/ড়/ঢ় kept (author-confirmed) | PASS |  |
| ব/ভ removed from HOMOPHONE_SETS (one of ten aspiration pairs; all-or-none) | PASS |  |
| N6_DROP_TARGETS == all matras + hasanta + chandrabindu | PASS |  |
| N6_ALTER_PAIRS chars are a subset of N6_DROP_TARGETS | PASS |  |
| N6_ALTER_PAIRS == {ি<->ী, ু<->ূ} only (M0 Q5 revised) | PASS |  |
| MATRA_TO_INDEPENDENT_VOWEL keys are all dependent vowel signs | PASS |  |
| MATRA_TO_INDEPENDENT_VOWEL values are all independent vowels | PASS |  |
| MATRA_TO_INDEPENDENT_VOWEL == {ু->উ, ূ->ঊ, ি->ই, ী->ঈ, ো->ও} only (restricted 2026-09-01) | PASS |  |
| HOMOPHONE_SETS_REJECTED includes the ten aspirated/unaspirated pairs | PASS |  |
| HOMOPHONE_SETS and HOMOPHONE_SETS_REJECTED are disjoint | PASS |  |
| every COMMON_CONJUNCTS entry is <consonant> HASANTA <consonant> | PASS |  |
| every string constant is NFC-idempotent | PASS |  |

## Structural code points

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| ্ | U+09CD | BENGALI SIGN VIRAMA |
| ‌ | U+200C | ZERO WIDTH NON-JOINER |
| ‍ | U+200D | ZERO WIDTH JOINER |
| ঁ | U+0981 | BENGALI SIGN CANDRABINDU |

## BANGLA_CONSONANTS (35)

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| ক | U+0995 | BENGALI LETTER KA |
| খ | U+0996 | BENGALI LETTER KHA |
| গ | U+0997 | BENGALI LETTER GA |
| ঘ | U+0998 | BENGALI LETTER GHA |
| ঙ | U+0999 | BENGALI LETTER NGA |
| চ | U+099A | BENGALI LETTER CA |
| ছ | U+099B | BENGALI LETTER CHA |
| জ | U+099C | BENGALI LETTER JA |
| ঝ | U+099D | BENGALI LETTER JHA |
| ঞ | U+099E | BENGALI LETTER NYA |
| ট | U+099F | BENGALI LETTER TTA |
| ঠ | U+09A0 | BENGALI LETTER TTHA |
| ড | U+09A1 | BENGALI LETTER DDA |
| ঢ | U+09A2 | BENGALI LETTER DDHA |
| ণ | U+09A3 | BENGALI LETTER NNA |
| ত | U+09A4 | BENGALI LETTER TA |
| থ | U+09A5 | BENGALI LETTER THA |
| দ | U+09A6 | BENGALI LETTER DA |
| ধ | U+09A7 | BENGALI LETTER DHA |
| ন | U+09A8 | BENGALI LETTER NA |
| প | U+09AA | BENGALI LETTER PA |
| ফ | U+09AB | BENGALI LETTER PHA |
| ব | U+09AC | BENGALI LETTER BA |
| ভ | U+09AD | BENGALI LETTER BHA |
| ম | U+09AE | BENGALI LETTER MA |
| য | U+09AF | BENGALI LETTER YA |
| র | U+09B0 | BENGALI LETTER RA |
| ল | U+09B2 | BENGALI LETTER LA |
| শ | U+09B6 | BENGALI LETTER SHA |
| ষ | U+09B7 | BENGALI LETTER SSA |
| স | U+09B8 | BENGALI LETTER SA |
| হ | U+09B9 | BENGALI LETTER HA |
| ড় | U+09A1 U+09BC | BENGALI LETTER DDA + BENGALI SIGN NUKTA |
| ঢ় | U+09A2 U+09BC | BENGALI LETTER DDHA + BENGALI SIGN NUKTA |
| য় | U+09AF U+09BC | BENGALI LETTER YA + BENGALI SIGN NUKTA |

## BANGLA_INDEPENDENT_VOWELS (11)

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| অ | U+0985 | BENGALI LETTER A |
| আ | U+0986 | BENGALI LETTER AA |
| ই | U+0987 | BENGALI LETTER I |
| ঈ | U+0988 | BENGALI LETTER II |
| উ | U+0989 | BENGALI LETTER U |
| ঊ | U+098A | BENGALI LETTER UU |
| ঋ | U+098B | BENGALI LETTER VOCALIC R |
| এ | U+098F | BENGALI LETTER E |
| ঐ | U+0990 | BENGALI LETTER AI |
| ও | U+0993 | BENGALI LETTER O |
| ঔ | U+0994 | BENGALI LETTER AU |

## MATRA_SIGNS (10)

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| া | U+09BE | BENGALI VOWEL SIGN AA |
| ি | U+09BF | BENGALI VOWEL SIGN I |
| ী | U+09C0 | BENGALI VOWEL SIGN II |
| ু | U+09C1 | BENGALI VOWEL SIGN U |
| ূ | U+09C2 | BENGALI VOWEL SIGN UU |
| ৃ | U+09C3 | BENGALI VOWEL SIGN VOCALIC R |
| ে | U+09C7 | BENGALI VOWEL SIGN E |
| ৈ | U+09C8 | BENGALI VOWEL SIGN AI |
| ো | U+09CB | BENGALI VOWEL SIGN O |
| ৌ | U+09CC | BENGALI VOWEL SIGN AU |

## BANGLA_SIGNS (4)

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| ঁ | U+0981 | BENGALI SIGN CANDRABINDU |
| ং | U+0982 | BENGALI SIGN ANUSVARA |
| ঃ | U+0983 | BENGALI SIGN VISARGA |
| ৎ | U+09CE | BENGALI LETTER KHANDA TA |

## BANGLA_DIGITS (10)

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| ০ | U+09E6 | BENGALI DIGIT ZERO |
| ১ | U+09E7 | BENGALI DIGIT ONE |
| ২ | U+09E8 | BENGALI DIGIT TWO |
| ৩ | U+09E9 | BENGALI DIGIT THREE |
| ৪ | U+09EA | BENGALI DIGIT FOUR |
| ৫ | U+09EB | BENGALI DIGIT FIVE |
| ৬ | U+09EC | BENGALI DIGIT SIX |
| ৭ | U+09ED | BENGALI DIGIT SEVEN |
| ৮ | U+09EE | BENGALI DIGIT EIGHT |
| ৯ | U+09EF | BENGALI DIGIT NINE |

## INSERTABLE_CHARS — N1 pool (49)

| glyph | code point(s) | class |
|---|---|---|
| ক | U+0995 | consonant |
| খ | U+0996 | consonant |
| গ | U+0997 | consonant |
| ঘ | U+0998 | consonant |
| ঙ | U+0999 | consonant |
| চ | U+099A | consonant |
| ছ | U+099B | consonant |
| জ | U+099C | consonant |
| ঝ | U+099D | consonant |
| ঞ | U+099E | consonant |
| ট | U+099F | consonant |
| ঠ | U+09A0 | consonant |
| ড | U+09A1 | consonant |
| ঢ | U+09A2 | consonant |
| ণ | U+09A3 | consonant |
| ত | U+09A4 | consonant |
| থ | U+09A5 | consonant |
| দ | U+09A6 | consonant |
| ধ | U+09A7 | consonant |
| ন | U+09A8 | consonant |
| প | U+09AA | consonant |
| ফ | U+09AB | consonant |
| ব | U+09AC | consonant |
| ভ | U+09AD | consonant |
| ম | U+09AE | consonant |
| য | U+09AF | consonant |
| র | U+09B0 | consonant |
| ল | U+09B2 | consonant |
| শ | U+09B6 | consonant |
| ষ | U+09B7 | consonant |
| স | U+09B8 | consonant |
| হ | U+09B9 | consonant |
| ড় | U+09A1 U+09BC | consonant |
| ঢ় | U+09A2 U+09BC | consonant |
| য় | U+09AF U+09BC | consonant |
| অ | U+0985 | independent_vowel |
| আ | U+0986 | independent_vowel |
| ই | U+0987 | independent_vowel |
| ঈ | U+0988 | independent_vowel |
| উ | U+0989 | independent_vowel |
| ঊ | U+098A | independent_vowel |
| ঋ | U+098B | independent_vowel |
| এ | U+098F | independent_vowel |
| ঐ | U+0990 | independent_vowel |
| ও | U+0993 | independent_vowel |
| ঔ | U+0994 | independent_vowel |
| ং | U+0982 | sign |
| ঃ | U+0983 | sign |
| ৎ | U+09CE | sign |

## IN_CLASS_POOLS — N3 substitution pools

| class | size | members |
|---|---|---|
| consonant | 35 | ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ ড় ঢ় য় |
| independent_vowel | 11 | অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ |
| digit | 10 | ০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯ |

## HOMOPHONE_SETS — N5 (8 sets)

| # | members | code points | note |
|---|---|---|---|
| 1 | শ / ষ / স | U+09B6 , U+09B7 , U+09B8 | letters |
| 2 | ন / ণ | U+09A8 , U+09A3 | letters |
| 3 | ই / ঈ | U+0987 , U+0988 | letters |
| 4 | উ / ঊ | U+0989 , U+098A | letters |
| 5 | ি / ী | U+09BF , U+09C0 | dependent signs |
| 6 | ু / ূ | U+09C1 , U+09C2 | dependent signs |
| 7 | র / ড় / ঢ় | U+09B0 , U+09A1 U+09BC , U+09A2 U+09BC | nukta letters |
| 8 | জ / য | U+099C , U+09AF | letters |

## HOMOPHONE_SETS_REJECTED — considered and rejected (14)

| members | code points |
|---|---|
| অ / আ | U+0985 , U+0986 |
| এ / ঐ | U+098F , U+0990 |
| ও / ঔ | U+0993 , U+0994 |
| ং / ঁ | U+0982 , U+0981 |
| ক / খ | U+0995 , U+0996 |
| গ / ঘ | U+0997 , U+0998 |
| চ / ছ | U+099A , U+099B |
| জ / ঝ | U+099C , U+099D |
| ট / ঠ | U+099F , U+09A0 |
| ড / ঢ | U+09A1 , U+09A2 |
| ত / থ | U+09A4 , U+09A5 |
| দ / ধ | U+09A6 , U+09A7 |
| প / ফ | U+09AA , U+09AB |
| ব / ভ | U+09AC , U+09AD |

## N6_DROP_TARGETS — N6 drop set (12)

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| া | U+09BE | BENGALI VOWEL SIGN AA |
| ি | U+09BF | BENGALI VOWEL SIGN I |
| ী | U+09C0 | BENGALI VOWEL SIGN II |
| ু | U+09C1 | BENGALI VOWEL SIGN U |
| ূ | U+09C2 | BENGALI VOWEL SIGN UU |
| ৃ | U+09C3 | BENGALI VOWEL SIGN VOCALIC R |
| ে | U+09C7 | BENGALI VOWEL SIGN E |
| ৈ | U+09C8 | BENGALI VOWEL SIGN AI |
| ো | U+09CB | BENGALI VOWEL SIGN O |
| ৌ | U+09CC | BENGALI VOWEL SIGN AU |
| ্ | U+09CD | BENGALI SIGN VIRAMA |
| ঁ | U+0981 | BENGALI SIGN CANDRABINDU |

## N6_ALTER_PAIRS — N6 may alter (else drop-only)

| a | b | code points |
|---|---|---|
| ি | ী | U+09BF <-> U+09C0 |
| ু | ূ | U+09C1 <-> U+09C2 |

## MATRA_TO_INDEPENDENT_VOWEL — N8 insert_vowel map

| matra | -> independent vowel | code points |
|---|---|---|
| ু | উ | U+09C1 -> U+0989 |
| ূ | ঊ | U+09C2 -> U+098A |
| ি | ই | U+09BF -> U+0987 |
| ী | ঈ | U+09C0 -> U+0988 |
| ো | ও | U+09CB -> U+0993 |

## COMMON_CONJUNCTS — N7 fixtures (25)

| glyph | components | code points |
|---|---|---|
| ক্ষ | ক + ্ + ষ | U+0995 U+09CD U+09B7 |
| জ্ঞ | জ + ্ + ঞ | U+099C U+09CD U+099E |
| ঞ্চ | ঞ + ্ + চ | U+099E U+09CD U+099A |
| ঞ্জ | ঞ + ্ + জ | U+099E U+09CD U+099C |
| ণ্ড | ণ + ্ + ড | U+09A3 U+09CD U+09A1 |
| ত্ত | ত + ্ + ত | U+09A4 U+09CD U+09A4 |
| ত্র | ত + ্ + র | U+09A4 U+09CD U+09B0 |
| দ্ধ | দ + ্ + ধ | U+09A6 U+09CD U+09A7 |
| দ্ব | দ + ্ + ব | U+09A6 U+09CD U+09AC |
| ন্ত | ন + ্ + ত | U+09A8 U+09CD U+09A4 |
| ন্দ | ন + ্ + দ | U+09A8 U+09CD U+09A6 |
| ন্ধ | ন + ্ + ধ | U+09A8 U+09CD U+09A7 |
| প্র | প + ্ + র | U+09AA U+09CD U+09B0 |
| ব্দ | ব + ্ + দ | U+09AC U+09CD U+09A6 |
| স্ত | স + ্ + ত | U+09B8 U+09CD U+09A4 |
| স্থ | স + ্ + থ | U+09B8 U+09CD U+09A5 |
| স্প | স + ্ + প | U+09B8 U+09CD U+09AA |
| শ্র | শ + ্ + র | U+09B6 U+09CD U+09B0 |
| ষ্ণ | ষ + ্ + ণ | U+09B7 U+09CD U+09A3 |
| ঙ্গ | ঙ + ্ + গ | U+0999 U+09CD U+0997 |
| ক্র | ক + ্ + র | U+0995 U+09CD U+09B0 |
| ক্ত | ক + ্ + ত | U+0995 U+09CD U+09A4 |
| ল্ল | ল + ্ + ল | U+09B2 U+09CD U+09B2 |
| ম্প | ম + ্ + প | U+09AE U+09CD U+09AA |
| ম্ব | ম + ্ + ব | U+09AE U+09CD U+09AC |

## CANDIDATE_CONJUNCTS_TRIPLE — excluded from N7 (3)

| glyph | code points |
|---|---|
| স্ত্র | U+09B8 U+09CD U+09A4 U+09CD U+09B0 |
| ন্ত্র | U+09A8 U+09CD U+09A4 U+09CD U+09B0 |
| ক্ষ্ম | U+0995 U+09CD U+09B7 U+09CD U+09AE |

## EMOJI_CODEPOINT_RANGES — N10 detection

| range | size | sample |
|---|---|---|
| U+1F300–U+1F5FF | 768 | CYCLONE |
| U+1F600–U+1F64F | 80 | GRINNING FACE |
| U+1F680–U+1F6FF | 128 | ROCKET |
| U+1F900–U+1F9FF | 256 | CIRCLED CROSS FORMEE WITH FOUR DOTS |
| U+1FA70–U+1FAFF | 144 | BALLET SHOES |
| U+2600–U+26FF | 256 | BLACK SUN WITH RAYS |
| U+2700–U+27BF | 192 | BLACK SAFETY SCISSORS |
| U+1F1E6–U+1F1FF | 26 | REGIONAL INDICATOR SYMBOL LETTER A |
| U+1F000–U+1F02F | 48 | MAHJONG TILE EAST WIND |

## EMOJI_SEQUENCE_GLUE — stripped with emoji in 'remove'

| glyph | code point(s) | Unicode name(s) |
|---|---|---|
| ‍ | U+200D | ZERO WIDTH JOINER |
| ️ | U+FE0F | VARIATION SELECTOR-16 |
| ︎ | U+FE0E | VARIATION SELECTOR-15 |
| ⃣ | U+20E3 | COMBINING ENCLOSING KEYCAP |

## Derived lookups (sizes)

| name | size | note |
|---|---|---|
| CONSONANT_SET | 35 |  |
| INDEPENDENT_VOWEL_SET | 11 |  |
| MATRA_SET | 10 |  |
| DIGIT_SET | 10 |  |
| CHAR_TO_CLASS | 56 | multi-codepoint keys: ড় ঢ় য় |
| COMBINING_MARKS | 16 |  |

