"""Constants"""

DB_VERSION = "7.4"

# Multilingual (.ltm) format: N strictly-1:1:N editions stored in one render-only
# file. Deliberately a DISTINCT version namespace from DB_VERSION so a .ltm is
# never probed as / migrated like a bilingual .lt (migrate_document_db is never
# called on it).
LTM_VERSION = "M1.0"

OPERATION_CALCULATE_CUSTOM = "calculate_custom"
OPERATION_CALCULATE_NEXT = "calculate_next"
OPERATION_RESOLVE = "resolve"
OPERATION_TRIVIAL = "trivial_alignment"
OPERATION_TRIVIAL_MULTI = "trivial_alignment_multi"
OPERATION_BUILD_LTM_FROM_PREPARED = "build_ltm_from_prepared"
OPERATION_ADD_LANGUAGE = "add_language"
