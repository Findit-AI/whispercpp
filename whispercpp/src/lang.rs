//! `Lang` — typed enum over whisper.cpp's supported languages, with
//! an `Other(SmolStr)` escape hatch for unknown ISO codes.

use crate::sys;
use smol_str::SmolStr;

/// Language code. Marked `#[non_exhaustive]` so new variants can be
/// added when whisper.cpp adds languages without forcing a
/// semver-major bump; carries an `Other(SmolStr)` variant so unknown
/// ISO codes flowing in from whisper's auto-detect don't fail an
/// indexing run.
///
/// **Canonicalisation invariant.** [`Lang::from_iso639_1`] maps known
/// codes to named variants and never produces `Other` for an
/// enum-known code. This keeps structural `PartialEq`/`Hash` correct:
/// `Lang::En != Lang::Other("en")` is fine because no API path
/// constructs `Lang::Other("en")`.
///
/// **Serde wire format.** Lowercase ISO-639-1 strings: `"en"`,
/// `"yue"`, etc. (a previous `derive(Serialize,
/// Deserialize)` produced Rust variant names like `"En"` and
/// `{"Other":"xx"}`, which contradicted documented config shapes
/// and made human-edited configs brittle. The custom impls
/// below canonicalise through [`Lang::from_iso639_1`] /
/// [`Lang::as_str`] so the in-memory representation stays as-is
/// while the wire format matches the docs.)
#[non_exhaustive]
#[allow(missing_docs)] // variants are ISO 639-1 codes; self-documenting by name
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Lang {
  En,
  Zh,
  De,
  Es,
  Ru,
  Ko,
  Fr,
  Ja,
  Pt,
  Tr,
  Pl,
  Ca,
  Nl,
  Ar,
  Sv,
  It,
  Id,
  Hi,
  Fi,
  Vi,
  He,
  Uk,
  El,
  Ms,
  Cs,
  Ro,
  Da,
  Hu,
  Ta,
  No,
  Th,
  Ur,
  Hr,
  Bg,
  Lt,
  La,
  Mi,
  Ml,
  Cy,
  Sk,
  Te,
  Fa,
  Lv,
  Bn,
  Sr,
  Az,
  Sl,
  Kn,
  Et,
  Mk,
  Br,
  Eu,
  Is,
  Hy,
  Ne,
  Mn,
  Bs,
  Kk,
  Sq,
  Sw,
  Gl,
  Mr,
  Pa,
  Si,
  Km,
  Sn,
  Yo,
  So,
  Af,
  Oc,
  Ka,
  Be,
  Tg,
  Sd,
  Gu,
  Am,
  Yi,
  Lo,
  Uz,
  Fo,
  Ht,
  Ps,
  Tk,
  Nn,
  Mt,
  Sa,
  Lb,
  My,
  Bo,
  Tl,
  Mg,
  As,
  Tt,
  Haw,
  Ln,
  Ha,
  Ba,
  Jw,
  Su,
  Yue,
  /// ISO 639-1 (or whisper-supplied) code that did not match any
  /// known variant. `from_iso639_1` and `as_str` round-trip
  /// through this for unknown codes; the indexer can log the
  /// SmolStr value and continue.
  Other(SmolStr),
}

impl Lang {
  /// Stable round-trip with [`Lang::from_iso639_1`]. Named variants
  /// emit their canonical lowercase ISO code; `Other(s)` emits `s`.
  #[inline]
  pub fn as_str(&self) -> &str {
    match self {
      Self::En => "en",
      Self::Zh => "zh",
      Self::De => "de",
      Self::Es => "es",
      Self::Ru => "ru",
      Self::Ko => "ko",
      Self::Fr => "fr",
      Self::Ja => "ja",
      Self::Pt => "pt",
      Self::Tr => "tr",
      Self::Pl => "pl",
      Self::Ca => "ca",
      Self::Nl => "nl",
      Self::Ar => "ar",
      Self::Sv => "sv",
      Self::It => "it",
      Self::Id => "id",
      Self::Hi => "hi",
      Self::Fi => "fi",
      Self::Vi => "vi",
      Self::He => "he",
      Self::Uk => "uk",
      Self::El => "el",
      Self::Ms => "ms",
      Self::Cs => "cs",
      Self::Ro => "ro",
      Self::Da => "da",
      Self::Hu => "hu",
      Self::Ta => "ta",
      Self::No => "no",
      Self::Th => "th",
      Self::Ur => "ur",
      Self::Hr => "hr",
      Self::Bg => "bg",
      Self::Lt => "lt",
      Self::La => "la",
      Self::Mi => "mi",
      Self::Ml => "ml",
      Self::Cy => "cy",
      Self::Sk => "sk",
      Self::Te => "te",
      Self::Fa => "fa",
      Self::Lv => "lv",
      Self::Bn => "bn",
      Self::Sr => "sr",
      Self::Az => "az",
      Self::Sl => "sl",
      Self::Kn => "kn",
      Self::Et => "et",
      Self::Mk => "mk",
      Self::Br => "br",
      Self::Eu => "eu",
      Self::Is => "is",
      Self::Hy => "hy",
      Self::Ne => "ne",
      Self::Mn => "mn",
      Self::Bs => "bs",
      Self::Kk => "kk",
      Self::Sq => "sq",
      Self::Sw => "sw",
      Self::Gl => "gl",
      Self::Mr => "mr",
      Self::Pa => "pa",
      Self::Si => "si",
      Self::Km => "km",
      Self::Sn => "sn",
      Self::Yo => "yo",
      Self::So => "so",
      Self::Af => "af",
      Self::Oc => "oc",
      Self::Ka => "ka",
      Self::Be => "be",
      Self::Tg => "tg",
      Self::Sd => "sd",
      Self::Gu => "gu",
      Self::Am => "am",
      Self::Yi => "yi",
      Self::Lo => "lo",
      Self::Uz => "uz",
      Self::Fo => "fo",
      Self::Ht => "ht",
      Self::Ps => "ps",
      Self::Tk => "tk",
      Self::Nn => "nn",
      Self::Mt => "mt",
      Self::Sa => "sa",
      Self::Lb => "lb",
      Self::My => "my",
      Self::Bo => "bo",
      Self::Tl => "tl",
      Self::Mg => "mg",
      Self::As => "as",
      Self::Tt => "tt",
      Self::Haw => "haw",
      Self::Ln => "ln",
      Self::Ha => "ha",
      Self::Ba => "ba",
      Self::Jw => "jw",
      Self::Su => "su",
      Self::Yue => "yue",
      Self::Other(s) => s.as_str(),
    }
  }
}

impl Lang {
  /// Total-function constructor: every `&str` produces a `Lang`.
  /// Known whisper.cpp codes canonicalise to their named variant;
  /// unknown codes go to `Lang::Other`. Never produces
  /// `Lang::Other("en")` for an enum-known code "en" — see the
  /// canonicalisation invariant on the type doc.
  pub fn from_iso639_1(s: &str) -> Self {
    match s {
      "en" | "En" | "eN" | "EN" => Self::En,
      "zh" | "Zh" | "zH" | "ZH" => Self::Zh,
      "de" | "De" | "dE" | "DE" => Self::De,
      "es" | "Es" | "eS" | "ES" => Self::Es,
      "ru" | "Ru" | "rU" | "RU" => Self::Ru,
      "ko" | "Ko" | "kO" | "KO" => Self::Ko,
      "fr" | "Fr" | "fR" | "FR" => Self::Fr,
      "ja" | "Ja" | "jA" | "JA" => Self::Ja,
      "pt" | "Pt" | "pT" | "PT" => Self::Pt,
      "tr" | "Tr" | "tR" | "TR" => Self::Tr,
      "pl" | "Pl" | "pL" | "PL" => Self::Pl,
      "ca" | "Ca" | "cA" | "CA" => Self::Ca,
      "nl" | "Nl" | "nL" | "NL" => Self::Nl,
      "ar" | "Ar" | "aR" | "AR" => Self::Ar,
      "sv" | "Sv" | "sV" | "SV" => Self::Sv,
      "it" | "It" | "iT" | "IT" => Self::It,
      "id" | "Id" | "iD" | "ID" => Self::Id,
      "hi" | "Hi" | "hI" | "HI" => Self::Hi,
      "fi" | "Fi" | "fI" | "FI" => Self::Fi,
      "vi" | "Vi" | "vI" | "VI" => Self::Vi,
      "he" | "He" | "hE" | "HE" => Self::He,
      "uk" | "Uk" | "uK" | "UK" => Self::Uk,
      "el" | "El" | "eL" | "EL" => Self::El,
      "ms" | "Ms" | "mS" | "MS" => Self::Ms,
      "cs" | "Cs" | "cS" | "CS" => Self::Cs,
      "ro" | "Ro" | "rO" | "RO" => Self::Ro,
      "da" | "Da" | "dA" | "DA" => Self::Da,
      "hu" | "Hu" | "hU" | "HU" => Self::Hu,
      "ta" | "Ta" | "tA" | "TA" => Self::Ta,
      "no" | "No" | "nO" | "NO" => Self::No,
      "th" | "Th" | "tH" | "TH" => Self::Th,
      "ur" | "Ur" | "uR" | "UR" => Self::Ur,
      "hr" | "Hr" | "hR" | "HR" => Self::Hr,
      "bg" | "Bg" | "bG" | "BG" => Self::Bg,
      "lt" | "Lt" | "lT" | "LT" => Self::Lt,
      "la" | "La" | "lA" | "LA" => Self::La,
      "mi" | "Mi" | "mI" | "MI" => Self::Mi,
      "ml" | "Ml" | "mL" | "ML" => Self::Ml,
      "cy" | "Cy" | "cY" | "CY" => Self::Cy,
      "sk" | "Sk" | "sK" | "SK" => Self::Sk,
      "te" | "Te" | "tE" | "TE" => Self::Te,
      "fa" | "Fa" | "fA" | "FA" => Self::Fa,
      "lv" | "Lv" | "lV" | "LV" => Self::Lv,
      "bn" | "Bn" | "bN" | "BN" => Self::Bn,
      "sr" | "Sr" | "sR" | "SR" => Self::Sr,
      "az" | "Az" | "aZ" | "AZ" => Self::Az,
      "sl" | "Sl" | "sL" | "SL" => Self::Sl,
      "kn" | "Kn" | "kN" | "KN" => Self::Kn,
      "et" | "Et" | "eT" | "ET" => Self::Et,
      "mk" | "Mk" | "mK" | "MK" => Self::Mk,
      "br" | "Br" | "bR" | "BR" => Self::Br,
      "eu" | "Eu" | "eU" | "EU" => Self::Eu,
      "is" | "Is" | "iS" | "IS" => Self::Is,
      "hy" | "Hy" | "hY" | "HY" => Self::Hy,
      "ne" | "Ne" | "nE" | "NE" => Self::Ne,
      "mn" | "Mn" | "mN" | "MN" => Self::Mn,
      "bs" | "Bs" | "bS" | "BS" => Self::Bs,
      "kk" | "Kk" | "kK" | "KK" => Self::Kk,
      "sq" | "Sq" | "sQ" | "SQ" => Self::Sq,
      "sw" | "Sw" | "sW" | "SW" => Self::Sw,
      "gl" | "Gl" | "gL" | "GL" => Self::Gl,
      "mr" | "Mr" | "mR" | "MR" => Self::Mr,
      "pa" | "Pa" | "pA" | "PA" => Self::Pa,
      "si" | "Si" | "sI" | "SI" => Self::Si,
      "km" | "Km" | "kM" | "KM" => Self::Km,
      "sn" | "Sn" | "sN" | "SN" => Self::Sn,
      "yo" | "Yo" | "yO" | "YO" => Self::Yo,
      "so" | "So" | "sO" | "SO" => Self::So,
      "af" | "Af" | "aF" | "AF" => Self::Af,
      "oc" | "Oc" | "oC" | "OC" => Self::Oc,
      "ka" | "Ka" | "kA" | "KA" => Self::Ka,
      "be" | "Be" | "bE" | "BE" => Self::Be,
      "tg" | "Tg" | "tG" | "TG" => Self::Tg,
      "sd" | "Sd" | "sD" | "SD" => Self::Sd,
      "gu" | "Gu" | "gU" | "GU" => Self::Gu,
      "am" | "Am" | "aM" | "AM" => Self::Am,
      "yi" | "Yi" | "yI" | "YI" => Self::Yi,
      "lo" | "Lo" | "lO" | "LO" => Self::Lo,
      "uz" | "Uz" | "uZ" | "UZ" => Self::Uz,
      "fo" | "Fo" | "fO" | "FO" => Self::Fo,
      "ht" | "Ht" | "hT" | "HT" => Self::Ht,
      "ps" | "Ps" | "pS" | "PS" => Self::Ps,
      "tk" | "Tk" | "tK" | "TK" => Self::Tk,
      "nn" | "Nn" | "nN" | "NN" => Self::Nn,
      "mt" | "Mt" | "mT" | "MT" => Self::Mt,
      "sa" | "Sa" | "sA" | "SA" => Self::Sa,
      "lb" | "Lb" | "lB" | "LB" => Self::Lb,
      "my" | "My" | "mY" | "MY" => Self::My,
      "bo" | "Bo" | "bO" | "BO" => Self::Bo,
      "tl" | "Tl" | "tL" | "TL" => Self::Tl,
      "mg" | "Mg" | "mG" | "MG" => Self::Mg,
      "as" | "As" | "aS" | "AS" => Self::As,
      "tt" | "Tt" | "tT" | "TT" => Self::Tt,
      "haw" | "Haw" | "hAW" | "HAW" => Self::Haw,
      "ln" | "Ln" | "lN" | "LN" => Self::Ln,
      "ha" | "Ha" | "hA" | "HA" => Self::Ha,
      "ba" | "Ba" | "bA" | "BA" => Self::Ba,
      "jw" | "Jw" | "jW" | "JW" => Self::Jw,
      "su" | "Su" | "sU" | "SU" => Self::Su,
      "yue" | "Yue" | "yUE" | "YUE" => Self::Yue,
      other => Self::Other(SmolStr::new(other)),
    }
  }

  /// Total-function constructor: every `&str` produces a `Lang`.
  /// Known whisper.cpp codes canonicalise to their named variant;
  /// unknown codes go to `Lang::Other`. Never produces
  /// `Lang::Other("en")` for an enum-known code "en" — see the
  /// canonicalisation invariant on the type doc.
  pub fn try_from_iso639_1(s: &str) -> Option<Self> {
    Some(match s {
      "en" | "En" | "eN" | "EN" => Self::En,
      "zh" | "Zh" | "zH" | "ZH" => Self::Zh,
      "de" | "De" | "dE" | "DE" => Self::De,
      "es" | "Es" | "eS" | "ES" => Self::Es,
      "ru" | "Ru" | "rU" | "RU" => Self::Ru,
      "ko" | "Ko" | "kO" | "KO" => Self::Ko,
      "fr" | "Fr" | "fR" | "FR" => Self::Fr,
      "ja" | "Ja" | "jA" | "JA" => Self::Ja,
      "pt" | "Pt" | "pT" | "PT" => Self::Pt,
      "tr" | "Tr" | "tR" | "TR" => Self::Tr,
      "pl" | "Pl" | "pL" | "PL" => Self::Pl,
      "ca" | "Ca" | "cA" | "CA" => Self::Ca,
      "nl" | "Nl" | "nL" | "NL" => Self::Nl,
      "ar" | "Ar" | "aR" | "AR" => Self::Ar,
      "sv" | "Sv" | "sV" | "SV" => Self::Sv,
      "it" | "It" | "iT" | "IT" => Self::It,
      "id" | "Id" | "iD" | "ID" => Self::Id,
      "hi" | "Hi" | "hI" | "HI" => Self::Hi,
      "fi" | "Fi" | "fI" | "FI" => Self::Fi,
      "vi" | "Vi" | "vI" | "VI" => Self::Vi,
      "he" | "He" | "hE" | "HE" => Self::He,
      "uk" | "Uk" | "uK" | "UK" => Self::Uk,
      "el" | "El" | "eL" | "EL" => Self::El,
      "ms" | "Ms" | "mS" | "MS" => Self::Ms,
      "cs" | "Cs" | "cS" | "CS" => Self::Cs,
      "ro" | "Ro" | "rO" | "RO" => Self::Ro,
      "da" | "Da" | "dA" | "DA" => Self::Da,
      "hu" | "Hu" | "hU" | "HU" => Self::Hu,
      "ta" | "Ta" | "tA" | "TA" => Self::Ta,
      "no" | "No" | "nO" | "NO" => Self::No,
      "th" | "Th" | "tH" | "TH" => Self::Th,
      "ur" | "Ur" | "uR" | "UR" => Self::Ur,
      "hr" | "Hr" | "hR" | "HR" => Self::Hr,
      "bg" | "Bg" | "bG" | "BG" => Self::Bg,
      "lt" | "Lt" | "lT" | "LT" => Self::Lt,
      "la" | "La" | "lA" | "LA" => Self::La,
      "mi" | "Mi" | "mI" | "MI" => Self::Mi,
      "ml" | "Ml" | "mL" | "ML" => Self::Ml,
      "cy" | "Cy" | "cY" | "CY" => Self::Cy,
      "sk" | "Sk" | "sK" | "SK" => Self::Sk,
      "te" | "Te" | "tE" | "TE" => Self::Te,
      "fa" | "Fa" | "fA" | "FA" => Self::Fa,
      "lv" | "Lv" | "lV" | "LV" => Self::Lv,
      "bn" | "Bn" | "bN" | "BN" => Self::Bn,
      "sr" | "Sr" | "sR" | "SR" => Self::Sr,
      "az" | "Az" | "aZ" | "AZ" => Self::Az,
      "sl" | "Sl" | "sL" | "SL" => Self::Sl,
      "kn" | "Kn" | "kN" | "KN" => Self::Kn,
      "et" | "Et" | "eT" | "ET" => Self::Et,
      "mk" | "Mk" | "mK" | "MK" => Self::Mk,
      "br" | "Br" | "bR" | "BR" => Self::Br,
      "eu" | "Eu" | "eU" | "EU" => Self::Eu,
      "is" | "Is" | "iS" | "IS" => Self::Is,
      "hy" | "Hy" | "hY" | "HY" => Self::Hy,
      "ne" | "Ne" | "nE" | "NE" => Self::Ne,
      "mn" | "Mn" | "mN" | "MN" => Self::Mn,
      "bs" | "Bs" | "bS" | "BS" => Self::Bs,
      "kk" | "Kk" | "kK" | "KK" => Self::Kk,
      "sq" | "Sq" | "sQ" | "SQ" => Self::Sq,
      "sw" | "Sw" | "sW" | "SW" => Self::Sw,
      "gl" | "Gl" | "gL" | "GL" => Self::Gl,
      "mr" | "Mr" | "mR" | "MR" => Self::Mr,
      "pa" | "Pa" | "pA" | "PA" => Self::Pa,
      "si" | "Si" | "sI" | "SI" => Self::Si,
      "km" | "Km" | "kM" | "KM" => Self::Km,
      "sn" | "Sn" | "sN" | "SN" => Self::Sn,
      "yo" | "Yo" | "yO" | "YO" => Self::Yo,
      "so" | "So" | "sO" | "SO" => Self::So,
      "af" | "Af" | "aF" | "AF" => Self::Af,
      "oc" | "Oc" | "oC" | "OC" => Self::Oc,
      "ka" | "Ka" | "kA" | "KA" => Self::Ka,
      "be" | "Be" | "bE" | "BE" => Self::Be,
      "tg" | "Tg" | "tG" | "TG" => Self::Tg,
      "sd" | "Sd" | "sD" | "SD" => Self::Sd,
      "gu" | "Gu" | "gU" | "GU" => Self::Gu,
      "am" | "Am" | "aM" | "AM" => Self::Am,
      "yi" | "Yi" | "yI" | "YI" => Self::Yi,
      "lo" | "Lo" | "lO" | "LO" => Self::Lo,
      "uz" | "Uz" | "uZ" | "UZ" => Self::Uz,
      "fo" | "Fo" | "fO" | "FO" => Self::Fo,
      "ht" | "Ht" | "hT" | "HT" => Self::Ht,
      "ps" | "Ps" | "pS" | "PS" => Self::Ps,
      "tk" | "Tk" | "tK" | "TK" => Self::Tk,
      "nn" | "Nn" | "nN" | "NN" => Self::Nn,
      "mt" | "Mt" | "mT" | "MT" => Self::Mt,
      "sa" | "Sa" | "sA" | "SA" => Self::Sa,
      "lb" | "Lb" | "lB" | "LB" => Self::Lb,
      "my" | "My" | "mY" | "MY" => Self::My,
      "bo" | "Bo" | "bO" | "BO" => Self::Bo,
      "tl" | "Tl" | "tL" | "TL" => Self::Tl,
      "mg" | "Mg" | "mG" | "MG" => Self::Mg,
      "as" | "As" | "aS" | "AS" => Self::As,
      "tt" | "Tt" | "tT" | "TT" => Self::Tt,
      "haw" | "Haw" | "hAW" | "HAW" => Self::Haw,
      "ln" | "Ln" | "lN" | "LN" => Self::Ln,
      "ha" | "Ha" | "hA" | "HA" => Self::Ha,
      "ba" | "Ba" | "bA" | "BA" => Self::Ba,
      "jw" | "Jw" | "jW" | "JW" => Self::Jw,
      "su" | "Su" | "sU" | "SU" => Self::Su,
      "yue" | "Yue" | "yUE" | "YUE" => Self::Yue,
      _ => return None,
    })
  }

  /// Human-readable English name for this language
  /// (`"english"` for `Lang::En`, `"chinese"` for `Lang::Zh`,
  /// etc.). Wraps `whisper_lang_str_full`, which reads from
  /// a static const `std::map<std::string, std::pair<int,
  /// std::string>> g_lang` inside whisper.cpp.
  ///
  /// Returns `None` when:
  /// * the variant is `Lang::Other(...)` with a code
  ///   whisper.cpp doesn't recognise;
  /// * the language code contains an interior NUL byte
  ///   (rejected at the [`lang_id_for`] CString conversion);
  /// * whisper.cpp's table returns NULL or non-UTF-8 (model
  ///   corruption / build issue).
  ///
  /// **Returns `Option<SmolStr>` (owned), not `&'static str`,
  /// despite `g_lang` being a static-storage object.** The
  /// pointer the C function hands us comes from
  /// `kv.second.second.c_str()` — the buffer is owned by a
  /// `std::string` member of `g_lang`, not an immortal C
  /// string literal. Treating that storage as Rust
  /// `'static` would let safe callers retain it across the
  /// `std::map`'s static-destructor cleanup at process exit
  /// (or — should the build ever switch to dynamic linking
  /// — across `dlclose`). Copying into an owned [`SmolStr`]
  /// (≤23 bytes inline; all whisper language names fit) ties
  /// the Rust lifetime to the caller's Rust ownership rather
  /// than to a C++ static destruction order we don't
  /// control.
  pub fn full_name(&self) -> Option<SmolStr> {
    let lang_id = lang_id_for(self.as_str())?;
    // SAFETY: pure C accessor reading from `g_lang`. We do
    // NOT retain the returned pointer past this function's
    // body — the `SmolStr::new` below copies the bytes
    // immediately.
    let raw = unsafe { sys::whisper_lang_str_full(lang_id) };
    if raw.is_null() {
      return None;
    }
    // SAFETY: NUL-terminated; valid for the duration of this
    // function call (the `g_lang` storage outlives any
    // running Rust code on the same thread). `to_bytes`
    // does not copy; we copy on the next line via
    // `SmolStr::new`, after which the C-side pointer is no
    // longer referenced.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    core::str::from_utf8(bytes).ok().map(SmolStr::new)
  }
}

/// Largest language id whisper.cpp recognises, equal to
/// `whisper_lang_max_id()`.
///
/// Useful when sizing the language-probability buffer for a
/// hypothetical future `whisper_lang_auto_detect` binding —
/// upstream's contract is "the array must be
/// `whisper_lang_max_id() + 1` in size".
pub fn lang_max_id() -> i32 {
  // SAFETY: pure C function reading a static const table size.
  unsafe { sys::whisper_lang_max_id() }
}

/// Look up the integer language id whisper.cpp uses internally
/// for a given short ISO-639-1 code (`"en"`) or English name
/// (`"english"`).
///
/// Returns `None` when:
/// * the input contains an interior NUL byte (rejected at the
///   `CString` conversion);
/// * whisper.cpp returns -1 ("unknown language");
/// * the shim caught a C++ exception. Upstream
///   `whisper_lang_id` does `g_lang.count(const char *)` /
///   `.at(const char *)` which constructs a temporary
///   `std::string` from the C string — both can throw
///   `std::bad_alloc` under memory pressure. The
///   `whispercpp_lang_id` shim catches and surfaces as a
///   `WHISPERCPP_ERR_*` sentinel; we collapse those to
///   `None` so the safe API never observes a thrown
///   exception across `extern "C"` (which would be UB).
///
/// Inverse of [`State::detected_lang`](crate::State::detected_lang)
/// at the integer-id level. Most callers should prefer the
/// typed [`Lang`] enum — this raw id is mainly useful for
/// building `whisper_token_lang(ctx, lang_id)` arguments
/// (see [`crate::Context::token_for_lang`]).
pub fn lang_id_for(name: &str) -> Option<i32> {
  // Defensive cap on input length. Whisper language entries
  // are short codes (`"en"` = 2 bytes) or English names
  // (longest is ~13 bytes for `"luxembourgish"`-class
  // entries); 32 bytes is a comfortable cap that keeps
  // `lang_id_for(&"x".repeat(N))`-style adversarial inputs
  // from reaching the upstream
  // `WHISPER_LOG_ERROR("unknown language '%s'", lang)` path.
  // That logger formats into a 1024-byte buffer and then
  // re-formats with the same `va_list` for long messages —
  // the `whispercpp-sys: log_internal va_copy` patch closes
  // the va-list-reuse UB structurally, but rejecting
  // obviously-not-a-language inputs at the safe-Rust
  // boundary is cheaper and avoids exercising the patched
  // path at all.
  let bytes = name.as_bytes();
  if bytes.len() > 32 {
    return None;
  }
  if bytes.contains(&0) {
    return None;
  }
  // Stack-only NUL-terminated buffer. Avoids the heap
  // allocation `CString::new(name)` would do per call.
  // 33 = 32 (cap) + 1 (NUL); zero-initialised so we don't
  // need an explicit terminator write.
  let mut buf = [0u8; 33];
  buf[..bytes.len()].copy_from_slice(bytes);
  let cstr_ptr: *const core::ffi::c_char = buf.as_ptr().cast();
  // SAFETY: buf is on the stack and outlives this call.
  // NUL-terminated by construction (zero-init array). The
  // shim wraps `whisper_lang_id` in try/catch so a
  // `std::bad_alloc` from the implicit `std::string(const
  // char *)` construction inside `g_lang.count()` cannot
  // unwind into Rust. Caught exceptions surface as
  // `WHISPERCPP_ERR_*` sentinels at -100..=-103, distinct
  // from the upstream "not found" sentinel of -1 — we
  // collapse both negative regions to `None`.
  let id = unsafe { sys::whispercpp_lang_id(cstr_ptr) };
  if id < 0 { None } else { Some(id) }
}

impl core::fmt::Display for Lang {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(self.as_str())
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
const _: () = {
  impl serde::Serialize for Lang {
    /// Serialize as the lowercase ISO-639-1 (or whisper-supplied)
    /// string code. Matches what [`Lang::as_str`] returns —
    /// `Lang::En` → `"en"`, `Lang::Other(SmolStr::new("xx"))` →
    /// `"xx"`. The previous `derive(Serialize)` produced Rust
    /// variant names like `"En"` and `{"Other":"xx"}`,
    /// contradicting the config docs.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
      S: serde::Serializer,
    {
      serializer.serialize_str(self.as_str())
    }
  }

  impl<'de> serde::Deserialize<'de> for Lang {
    /// Deserialize from an ISO-639-1 string code, **case-insensitive**.
    ///
    /// Accepts any ASCII-letter case (`"en"`, `"EN"`, `"En"`,
    /// `"eN"` all canonicalise to `Lang::En`); whisper.cpp's
    /// language codes are conventionally lowercase but the ISO
    /// standard treats them as case-insensitive, and human-edited
    /// configs naturally use mixed case. The accepted alphabet
    /// after lowercasing is `[a-z]{1,8}` — matches the
    /// alignment-stage validation in `runner/whisper_pool.rs`'s
    /// `validate_language_code` so an "EN" config
    /// produces a Lang that the FFI layer happily accepts.
    ///
    /// Routes through [`Lang::from_iso639_1`] *after* lowercasing
    /// so input matching a named variant canonicalises to that
    /// variant rather than landing in `Other`. Unknown codes pass
    /// through `Lang::Other(SmolStr::new(lowered))` — the inner
    /// string is always lowercase, preserving the canonicalisation
    /// invariant across the serde boundary AND keeping the
    /// language-string intern table bounded.
    ///
    /// Round-trip asymmetry note: `"EN"` deserialises to
    /// `Lang::En` which then *serialises* as `"en"`. This is
    /// intentional — the on-disk canonical form is lowercase.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
      D: serde::Deserializer<'de>,
    {
      use serde::de::Error as _;

      let s = <&str as serde::Deserialize>::deserialize(deserializer)?;
      if s.is_empty() {
        return Err(D::Error::custom("Lang code is empty"));
      }
      if s.len() > 8 {
        return Err(D::Error::custom(format!(
          "Lang code longer than 8 bytes ({} bytes); whisper.cpp codes are 2-3 ASCII letters",
          s.len()
        )));
      }
      if !s.bytes().all(|b| b.is_ascii_alphabetic()) {
        return Err(D::Error::custom(
          "Lang code must be ASCII letters [a-zA-Z] only (no digits, dashes, or non-ASCII)",
        ));
      }
      // Avoid the lowercasing allocation when input is already canonical.
      if s.bytes().all(|b| b.is_ascii_lowercase()) {
        Ok(Lang::from_iso639_1(s))
      } else {
        use smol_str::StrExt;

        let lowered = s.to_ascii_lowercase_smolstr();
        Ok(Lang::try_from_iso639_1(&lowered).unwrap_or(Self::Other(lowered)))
      }
    }
  }
};

#[cfg(test)]
mod tests {
  use super::*;

  /// Every named variant round-trips through `from_iso639_1(as_str)`
  /// AND does not match `Lang::Other(_)`. This is the
  /// canonicalisation invariant.
  #[test]
  fn named_variants_canonicalise() {
    let known = [
      Lang::En,
      Lang::Zh,
      Lang::De,
      Lang::Es,
      Lang::Ru,
      Lang::Ko,
      Lang::Fr,
      Lang::Ja,
      Lang::Pt,
      Lang::Tr,
      Lang::Pl,
      Lang::Ca,
      Lang::Nl,
      Lang::Ar,
      Lang::Sv,
      Lang::It,
      Lang::Id,
      Lang::Hi,
      Lang::Fi,
      Lang::Vi,
      Lang::He,
      Lang::Uk,
      Lang::El,
      Lang::Ms,
      Lang::Cs,
      Lang::Ro,
      Lang::Da,
      Lang::Hu,
      Lang::Ta,
      Lang::No,
      Lang::Th,
      Lang::Ur,
      Lang::Hr,
      Lang::Bg,
      Lang::Lt,
      Lang::La,
      Lang::Mi,
      Lang::Ml,
      Lang::Cy,
      Lang::Sk,
      Lang::Te,
      Lang::Fa,
      Lang::Lv,
      Lang::Bn,
      Lang::Sr,
      Lang::Az,
      Lang::Sl,
      Lang::Kn,
      Lang::Et,
      Lang::Mk,
      Lang::Br,
      Lang::Eu,
      Lang::Is,
      Lang::Hy,
      Lang::Ne,
      Lang::Mn,
      Lang::Bs,
      Lang::Kk,
      Lang::Sq,
      Lang::Sw,
      Lang::Gl,
      Lang::Mr,
      Lang::Pa,
      Lang::Si,
      Lang::Km,
      Lang::Sn,
      Lang::Yo,
      Lang::So,
      Lang::Af,
      Lang::Oc,
      Lang::Ka,
      Lang::Be,
      Lang::Tg,
      Lang::Sd,
      Lang::Gu,
      Lang::Am,
      Lang::Yi,
      Lang::Lo,
      Lang::Uz,
      Lang::Fo,
      Lang::Ht,
      Lang::Ps,
      Lang::Tk,
      Lang::Nn,
      Lang::Mt,
      Lang::Sa,
      Lang::Lb,
      Lang::My,
      Lang::Bo,
      Lang::Tl,
      Lang::Mg,
      Lang::As,
      Lang::Tt,
      Lang::Haw,
      Lang::Ln,
      Lang::Ha,
      Lang::Ba,
      Lang::Jw,
      Lang::Su,
      Lang::Yue,
    ];
    assert_eq!(
      known.len(),
      100,
      "must keep the 100-variant Appendix C list in sync"
    );
    for v in known.iter() {
      let round = Lang::from_iso639_1(v.as_str());
      assert_eq!(&round, v, "round-trip failed for {:?}", v);
      assert!(
        !matches!(round, Lang::Other(_)),
        "{:?} canonicalised to Other; this breaks Eq/Hash",
        v
      );
    }
  }

  #[test]
  fn unknown_codes_land_in_other() {
    let r = Lang::from_iso639_1("zzz");
    assert_eq!(r, Lang::Other(SmolStr::new("zzz")));
    assert_eq!(r.as_str(), "zzz");
  }

  #[test]
  fn other_round_trips_via_as_str() {
    let r = Lang::Other(SmolStr::new("xx"));
    assert_eq!(r.as_str(), "xx");
    assert_eq!(Lang::from_iso639_1(r.as_str()), r);
  }

  // --- custom serde wire format ---

  #[cfg(feature = "serde")]
  #[test]
  fn serde_named_variant_serializes_as_lowercase_iso() {
    let json = serde_json::to_string(&Lang::En).expect("serialize");
    assert_eq!(
      json, "\"en\"",
      "Lang::En must serialize as \"en\", not \"En\""
    );
    let json = serde_json::to_string(&Lang::Yue).expect("serialize");
    assert_eq!(json, "\"yue\"");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_other_serializes_as_inner_string() {
    let v = Lang::Other(SmolStr::new("xx"));
    let json = serde_json::to_string(&v).expect("serialize");
    assert_eq!(
      json, "\"xx\"",
      "Lang::Other(\"xx\") must serialize as \"xx\""
    );
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_named_variant_round_trips() {
    let json = "\"en\"";
    let lang: Lang = serde_json::from_str(json).expect("deserialize");
    assert_eq!(lang, Lang::En);
    // Re-serialize and verify identical wire form.
    assert_eq!(serde_json::to_string(&lang).unwrap(), json);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_unknown_iso_code_round_trips_via_other() {
    let json = "\"xx\"";
    let lang: Lang = serde_json::from_str(json).expect("deserialize");
    assert_eq!(lang, Lang::Other(SmolStr::new("xx")));
    assert_eq!(serde_json::to_string(&lang).unwrap(), json);
  }

  /// Canonicalisation invariant must hold across serde:
  /// deserializing a code that matches a named variant lands in
  /// the named variant, not in `Other`.
  #[cfg(feature = "serde")]
  #[test]
  fn serde_deserializes_known_codes_to_named_variants() {
    let lang: Lang = serde_json::from_str("\"en\"").unwrap();
    assert!(matches!(lang, Lang::En), "must canonicalise to Lang::En");
    let lang: Lang = serde_json::from_str("\"yue\"").unwrap();
    assert!(matches!(lang, Lang::Yue));
  }

  /// Case-insensitive deserialization (UX win — users editing
  /// configs naturally use mixed case): `"EN"`, `"En"`, `"eN"`,
  /// `"en"` all canonicalise to `Lang::En`. The on-disk
  /// canonical form is lowercase (so re-serialization always
  /// emits `"en"`), but reading is permissive.
  #[cfg(feature = "serde")]
  #[test]
  fn serde_accepts_any_case_for_named_variant() {
    for input in ["\"en\"", "\"EN\"", "\"En\"", "\"eN\""] {
      let lang: Lang = serde_json::from_str(input).expect(input);
      assert_eq!(
        lang,
        Lang::En,
        "input {input} must canonicalise to Lang::En"
      );
      // Re-serialisation always emits the lowercase form.
      assert_eq!(serde_json::to_string(&lang).unwrap(), "\"en\"");
    }
  }

  /// Mixed-case unknown codes also canonicalise — `"XX"`
  /// deserialises to `Lang::Other(SmolStr::new("xx"))`,
  /// preserving the canonicalisation invariant (no
  /// `Lang::Other("XX")` ever exists in the type).
  #[cfg(feature = "serde")]
  #[test]
  fn serde_lowercases_unknown_code_into_other() {
    let lang: Lang = serde_json::from_str("\"XX\"").expect("deserialize");
    assert_eq!(lang, Lang::Other(SmolStr::new("xx")));
    let lang: Lang = serde_json::from_str("\"Xx\"").expect("deserialize");
    assert_eq!(lang, Lang::Other(SmolStr::new("xx")));
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_empty_string() {
    let res: Result<Lang, _> = serde_json::from_str("\"\"");
    assert!(res.is_err());
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_overlong_code() {
    let res: Result<Lang, _> = serde_json::from_str("\"abcdefghi\"");
    assert!(res.is_err(), "9-byte code must be rejected");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_non_ascii_letters() {
    let res: Result<Lang, _> = serde_json::from_str("\"français\"");
    assert!(res.is_err(), "non-ASCII must be rejected");
    let res: Result<Lang, _> = serde_json::from_str("\"a-b\"");
    assert!(res.is_err(), "dash must be rejected");
    let res: Result<Lang, _> = serde_json::from_str("\"a1b\"");
    assert!(res.is_err(), "digits must be rejected");
  }

  /// Old derive-shaped JSON for `Other` (`{"Other":"xx"}`) must
  /// fail with the new custom impl — it's an externally-tagged
  /// object, not a string. Documents the breaking wire-format
  /// change for migrators.
  ///
  /// Note: legacy `"En"` (Rust variant name) is now ACCEPTED as
  /// a side-effect of case-insensitive deserialization. That's a
  /// happy accident for migration — old configs that happened to
  /// use the variant-name form continue to work, just with the
  /// canonical lowercase form on round-trip. No special handling
  /// needed.
  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_legacy_other_as_map() {
    let res: Result<Lang, _> = serde_json::from_str(r#"{"Other":"xx"}"#);
    assert!(
      res.is_err(),
      "legacy Other-as-map encoding must be rejected"
    );
  }

  /// `lang_max_id()` is the size of whisper.cpp's static
  /// `g_lang` table minus 1. The current bundled whisper
  /// supports 99 languages (ids 0..=98), so the max id is 98.
  /// Pin so an upstream rebuild that adds languages doesn't
  /// silently change buffer-sizing assumptions in callers.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: calls whisper_lang_max_id")]
  fn lang_max_id_matches_known_table_size() {
    let max_id = lang_max_id();
    assert!(
      max_id >= 98,
      "lang_max_id should cover at least the v1.8.4 table (98); got {max_id}"
    );
    // Sanity upper bound: there are <200 ISO 639-1 codes total;
    // a value drastically above that means the table was
    // corrupted or our binding is off.
    assert!(max_id < 200, "lang_max_id={max_id} is implausibly large");
  }

  /// `lang_id_for` round-trips with `Lang::as_str()` for every
  /// named variant (the FFI's `whisper_lang_id` accepts both
  /// short codes and English names; we feed it short codes).
  #[test]
  #[cfg_attr(miri, ignore = "FFI: calls whispercpp_lang_id")]
  fn lang_id_for_round_trips_named_variants() {
    for lang in [Lang::En, Lang::Zh, Lang::Ja, Lang::Ko, Lang::Yue] {
      let id = lang_id_for(lang.as_str())
        .unwrap_or_else(|| panic!("lang_id_for({}) returned None", lang.as_str()));
      assert!(
        id >= 0,
        "lang_id_for({}) returned negative {id}",
        lang.as_str()
      );
      assert!(
        id <= lang_max_id(),
        "lang_id_for({}) = {id} exceeds lang_max_id = {}",
        lang.as_str(),
        lang_max_id()
      );
    }
  }

  /// `lang_id_for` returns `None` for codes whisper.cpp
  /// doesn't recognise (covers `Lang::Other(...)` ISO codes
  /// like `"xx"` that aren't in the upstream table).
  #[test]
  #[cfg_attr(miri, ignore = "FFI: calls whispercpp_lang_id")]
  fn lang_id_for_returns_none_on_unknown() {
    assert_eq!(lang_id_for("xx"), None);
    assert_eq!(lang_id_for("definitely-not-a-language"), None);
  }

  /// `lang_id_for` rejects strings with interior NUL bytes at
  /// the `CString` conversion (would otherwise pass to whisper
  /// as a truncated lookup; explicit `None` is clearer).
  #[test]
  fn lang_id_for_rejects_interior_nul() {
    assert_eq!(lang_id_for("en\0extra"), None);
  }

  /// Adversarially-long inputs to `lang_id_for` are rejected
  /// at the safe-Rust boundary BEFORE reaching the upstream
  /// `WHISPER_LOG_ERROR("unknown language '%s'", ...)` log
  /// path. The companion `whispercpp-sys: log_internal
  /// va_copy` patch closes the va-list-reuse UB structurally,
  /// but rejecting obviously-not-a-language inputs at the
  /// boundary is cheaper. The cap (32 bytes) gives generous
  /// headroom over the longest known whisper language name
  /// (`"luxembourgish"`-class entries, ~13 bytes).
  #[test]
  #[cfg_attr(miri, ignore = "FFI: just-under-cap branch reaches whispercpp_lang_id")]
  fn lang_id_for_rejects_overlong_input() {
    let long = "x".repeat(2000);
    assert_eq!(lang_id_for(&long), None);
    // One byte over the cap is rejected at the safe-Rust boundary
    // (the cap is `> 32`, so `len() == 33` fails the check).
    let one_over_cap = "x".repeat(33);
    assert_eq!(lang_id_for(&one_over_cap), None);
    // Exactly at the cap (`len() == 32`) passes the boundary check
    // and reaches FFI; whisper.cpp returns "unknown" for
    // non-language strings, which we surface as None.
    let at_cap = "x".repeat(32);
    assert_eq!(lang_id_for(&at_cap), None);
  }

  /// `Lang::full_name()` returns the English name for known
  /// languages — `"english"` for `Lang::En`, `"chinese"` for
  /// `Lang::Zh`, etc. Owned `SmolStr` (not `&'static str`)
  /// so the result doesn't outlive the C++ static
  /// `g_lang.std::string` storage; see the function's
  /// doc-comment for the soundness rationale.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: calls whisper_lang_str_full")]
  fn full_name_returns_english_name_for_known_langs() {
    assert_eq!(Lang::En.full_name(), Some(SmolStr::new("english")));
    assert_eq!(Lang::Zh.full_name(), Some(SmolStr::new("chinese")));
    assert_eq!(Lang::Ja.full_name(), Some(SmolStr::new("japanese")));
    assert_eq!(Lang::Fr.full_name(), Some(SmolStr::new("french")));
  }

  /// `Lang::full_name()` returns `None` for `Lang::Other(...)`
  /// codes whisper.cpp doesn't recognise.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: lang_id_for(\"xx\") reaches whispercpp_lang_id")]
  fn full_name_returns_none_for_unknown_other() {
    let unknown = Lang::Other(SmolStr::new("xx"));
    assert_eq!(unknown.full_name(), None);
  }
}
