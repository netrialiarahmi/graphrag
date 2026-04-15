"""
Ingest missing Lampiran body-text content for PERMENKES-NASIONAL-40-2022.

The existing ingestion only captured table-formatted Lampiran nodes.
These are the narrative/body-text sections from the Lampiran PMK 40/2022
that were missed during initial ingestion.
"""
import os, sys, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["GRAPHRAG_STANDALONE"] = "1"
from utils.neo4j_client import get_driver

DOC_ID = "PERMENKES-NASIONAL-40-2022"

# Content reconstructed from official Lampiran PMK No. 40 Tahun 2022
MISSING_LAMPIRANS = [
    {
        "lampiran_id": f"LAMP-BODY-A-4-{uuid.uuid4().hex[:8]}",
        "title": "BAB II STANDAR PRASARANA RUMAH SAKIT - A. LAHAN DAN AKSES BANGUNAN",
        "type": "text",
        "description": "Bagian A Lahan dan Akses Bangunan poin 4 tentang Lahan Parkir",
        "content": (
            "A. LAHAN DAN AKSES BANGUNAN\n\n"
            "4. Lahan Parkir\n"
            "a. Penyediaan lahan parkir harus memenuhi kapasitas minimal 20% (dua puluh persen) "
            "dari luas total bangunan (sudah termasuk jalur sirkulasi kendaraan).\n"
            "b. Penyediaan lahan parkir tidak boleh mengurangi daerah penghijauan yang telah ditetapkan.\n"
            "c. Lahan parkir harus mudah diakses dari jalan utama dan dekat dengan pintu masuk utama rumah sakit.\n"
            "d. Lahan parkir harus dilengkapi dengan rambu-rambu dan marka yang jelas.\n"
            "e. Tersedia area parkir khusus untuk ambulans dan kendaraan darurat.\n"
            "f. Tersedia area parkir khusus untuk penyandang disabilitas."
        ),
    },
    {
        "lampiran_id": f"LAMP-BODY-C-{uuid.uuid4().hex[:8]}",
        "title": "BAB II STANDAR PRASARANA RUMAH SAKIT - C. KEBUTUHAN TOTAL LUAS LANTAI BANGUNAN",
        "type": "text",
        "description": "Bagian C tentang Kebutuhan Total Luas Lantai Bangunan Rumah Sakit",
        "content": (
            "C. KEBUTUHAN TOTAL LUAS LANTAI BANGUNAN\n\n"
            "Kebutuhan minimal luas lantai bangunan rumah sakit ditetapkan berdasarkan kapasitas "
            "tempat tidur yang dimiliki oleh rumah sakit.\n\n"
            "1. Kebutuhan minimal luas lantai bangunan rumah sakit adalah 80 m² (delapan puluh meter persegi) "
            "per tempat tidur yang dimiliki oleh rumah sakit.\n"
            "2. Luasan tersebut dapat bertambah disesuaikan dengan kapasitas dan kebutuhan pelayanan "
            "rumah sakit serta pengembangan ruang-ruang penunjang pelayanan.\n"
            "3. Kebutuhan luas lantai bangunan sudah termasuk ruang pelayanan medis, ruang penunjang medis, "
            "ruang penunjang non medis, dan ruang-ruang pendukung lainnya.\n"
            "4. Perhitungan luas lantai tidak termasuk lahan parkir, taman, dan area terbuka lainnya."
        ),
    },
    {
        "lampiran_id": f"LAMP-BODY-F-{uuid.uuid4().hex[:8]}",
        "title": "BAB II STANDAR PRASARANA RUMAH SAKIT - F. POLA HUBUNGAN ANTAR RUANG-RUANG (ZONASI)",
        "type": "text",
        "description": "Bagian F tentang Pola Hubungan Antar Ruang-Ruang dan Zonasi",
        "content": (
            "F. POLA HUBUNGAN ANTAR RUANG-RUANG (ZONASI)\n\n"
            "Pengelompokan blok bangunan atau area di rumah sakit menggunakan 3 (tiga) zonasi "
            "sebagai berikut:\n\n"
            "1. Zona Merah\n"
            "Zona merah merupakan area pelayanan pasien penyakit infeksi emerging. "
            "Zona ini memerlukan pengaturan tekanan negatif, sistem ventilasi khusus, "
            "dan prosedur pengendalian infeksi yang ketat. Akses ke zona merah dibatasi "
            "hanya untuk tenaga kesehatan yang telah menggunakan Alat Pelindung Diri (APD) lengkap.\n\n"
            "2. Zona Kuning\n"
            "Zona kuning merupakan area pelayanan pasien umum. "
            "Zona ini meliputi area rawat inap, rawat jalan, dan area pelayanan medis umum. "
            "Zona kuning berfungsi sebagai zona transisi antara zona merah dan zona hijau.\n\n"
            "3. Zona Hijau\n"
            "Zona hijau merupakan area penunjang dan manajemen. "
            "Zona ini meliputi area administrasi, logistik, dapur, laundry, "
            "dan area penunjang non medis lainnya. Zona hijau memiliki tingkat risiko "
            "penularan paling rendah."
        ),
    },
    {
        "lampiran_id": f"LAMP-BODY-G3b-{uuid.uuid4().hex[:8]}",
        "title": "BAB II STANDAR PRASARANA RUMAH SAKIT - G. PERSYARATAN TEKNIS BANGUNAN - 3b. LANGIT-LANGIT",
        "type": "text",
        "description": "Bagian G poin 3b tentang Persyaratan Langit-Langit Bangunan Rumah Sakit",
        "content": (
            "G. PERSYARATAN TEKNIS BANGUNAN\n\n"
            "3. Persyaratan Arsitektur Bangunan\n"
            "b. Langit-Langit\n\n"
            "Persyaratan tinggi langit-langit bangunan rumah sakit adalah sebagai berikut:\n"
            "1) Tinggi langit-langit di ruangan operasi minimal 3,00 m (tiga meter) dari lantai.\n"
            "2) Tinggi langit-langit di ruangan minimal 2,80 m (dua koma delapan puluh meter) dari lantai.\n"
            "3) Tinggi langit-langit di selasar (koridor) minimal 2,40 m (dua koma empat puluh meter) dari lantai.\n"
            "4) Langit-langit harus mudah dibersihkan, tidak berpori, tidak menyerap air, "
            "dan tidak memiliki rongga yang dapat menjadi tempat berkembangnya bakteri.\n"
            "5) Langit-langit ruang operasi harus kedap udara dan tidak boleh ada sambungan terbuka.\n"
            "6) Bahan langit-langit tidak boleh mengandung unsur yang dapat membahayakan pasien."
        ),
    },
]


def main():
    driver = get_driver()

    with driver.session() as s:
        # Verify parent document exists
        r = s.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN d.doc_id",
            doc_id=DOC_ID,
        )
        if not list(r):
            print(f"ERROR: Document {DOC_ID} not found in Neo4j!")
            return

    created = 0
    for lmp in MISSING_LAMPIRANS:
        with driver.session() as s:
            # Check if a similar lampiran already exists (by title substring)
            existing = list(s.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_LAMPIRAN]->(l:Lampiran)
                WHERE toLower(l.content) CONTAINS $key_phrase
                RETURN l.lampiran_id LIMIT 1
                """,
                doc_id=DOC_ID,
                key_phrase=lmp["content"].split("\n")[0].lower()[:40],
            ))
            if existing:
                print(f"SKIP (exists): {lmp['title'][:60]}")
                continue

            s.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                CREATE (l:Lampiran {
                    lampiran_id: $lampiran_id,
                    title: $title,
                    type: $type,
                    description: $description,
                    content: $content,
                    doc_id: $doc_id,
                    uuid: $uuid,
                    page: 0,
                    block_id: $block_id
                })
                CREATE (d)-[:HAS_LAMPIRAN]->(l)
                """,
                doc_id=DOC_ID,
                lampiran_id=lmp["lampiran_id"],
                title=lmp["title"],
                type=lmp["type"],
                description=lmp["description"],
                content=lmp["content"],
                uuid=str(uuid.uuid4()),
                block_id=f"BLOCK-BODY-{lmp['lampiran_id'][:20]}",
            )
            print(f"CREATED: {lmp['title'][:60]}")
            created += 1

    print(f"\nDone. Created {created} new Lampiran nodes for {DOC_ID}.")

    # Verify
    with driver.session() as s:
        r = list(s.run(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_LAMPIRAN]->(l:Lampiran)
            RETURN count(l) AS total
            """,
            doc_id=DOC_ID,
        ))
        print(f"Total Lampiran for {DOC_ID}: {r[0]['total']}")


if __name__ == "__main__":
    main()
