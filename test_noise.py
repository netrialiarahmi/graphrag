import re

def is_noise(content: str) -> bool:
    content = content.lower().strip()
    if not content or content in ["cukup jelas.", "cukup jelas", "(kosong)", "-"]:
        return True
    
    # Strip common legislative prefix markers
    cleaned = re.sub(r'(?:pasal\s+\d+|ayat\s*\(\d+\)|huruf\s+[a-z]|angka\s+\d+)', '', content, flags=re.IGNORECASE)
    
    # Strip non-alphanumeric
    cleaned_no_space = re.sub(r'[\s\.\-\(\)]+', '', cleaned)
    
    # If the only text left is "cukupjelas" repeated, it's noise
    cukup_jelas_pattern = r'^(cukupjelas)+$'
    
    if re.match(cukup_jelas_pattern, cleaned_no_space):
        return True
        
    return False

samples = [
    "Cukup jelas.",
    "ayat (1)\nhuruf a\ncukup jelas.\nhuruf b\ncukup jelas.",
    "Pasal 1\nayat (1) cukup jelas.",
    "ayat (2)\ncukup jelas.",
    "huruf a\nini penjelasan penting",
    "ayat (1)\npenentuan jumlah anggota dprd provinsi",
    "(kosong)",
    "   "
]

for s in samples:
    print(f"[{is_noise(s)}] : {repr(s)}")