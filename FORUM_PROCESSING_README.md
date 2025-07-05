# Forum Data Processing Scripts

สคริปต์สำหรับประมวลผลข้อมูลกระทู้ถามแพทย์จากไฟล์ในโฟลเดอร์ `data_scraped_txt/` และ upsert ขึ้น Pinecone

## 📁 ไฟล์หลักที่ใช้งาน

### `pinecone_batch_upsert.py` (ไฟล์หลักที่ใช้งาน)
- **ประมวลผลหลายไฟล์** - รองรับการประมวลผลไฟล์หลายไฟล์ใน `data_scraped_txt/`
- **Batch Processing** - ประมวลผลแบบ batch ขนาด 20 documents ต่อครั้ง
- **Smart Deduplication** - ป้องกันการ upsert ข้อมูลซ้ำ
- **Content Storage** - เก็บ content ใน metadata เพื่อการ retrieve
- **Error Handling** - จัดการ error และ logging ครบถ้วน
- **Summary Generation** - สร้างไฟล์สรุปการประมวลผล

## 🚀 วิธีการใช้งาน

### วิธีหลัก: ใช้ Batch Processor
```bash
python pinecone_batch_upsert.py
```

### การตั้งค่าเบื้องต้น
1. วางไฟล์ forum data ในโฟลเดอร์ `data_scraped_txt/`
2. ตั้งค่า environment variables หรือ Streamlit secrets
3. รันสคริปต์

### ผลลัพธ์ที่ได้
- ข้อมูลถูก upsert ขึ้น Pinecone
- ไฟล์สรุป `upserted_data/batch_processing_summary.json`
- Console output แสดงความคืบหน้า

## 📊 Metadata ที่สร้างขึ้น

`pinecone_batch_upsert.py` สร้าง metadata ดังนี้:

```json
{
  "source": "https://www.agnoshealth.com/forums/เสี่ยงต่อโรคซึมเศร้า/2675",
  "info": "https://www.agnoshealth.com/assets/icons/thumbs-up/thumbs-up-filled-blue.svg)5![thumbs-up](https://www.agnoshealth.com/assets/icons/forum/comment-checked-filled-blue.svg)แพทย์ตอบคำปรึกษาแล้ว](https://www.agnoshealth.com/forums/เสี่ยงต่อโรคซึมเศร้า/2675)![thumbs-up](https://www.agnoshealth.com/assets/icons/forum/share-outline-blue.svg)แชร์กระทู้",
  "forum_posted_timestamp": "8/22/2022",
  "forum_reply_count": 3.0,
  "scraped_at": "2025-07-06T00:50:27.127031",
  "record_index": 3.0,
  "content": "สวัสดีค่ะ อยากจะปรึกษาค่ะในช่วง2-3วันที่ผ่านมามีความรู้สึกดาวน์มากๆ..."
}
```

### ฟิลด์สำคัญ:
- **source**: URL ที่สะอาดของ forum post
- **content**: เนื้อหาของ forum post (เก็บใน metadata เพื่อการ retrieve)
- **forum_posted_timestamp**: วันที่โพสต์
- **forum_reply_count**: จำนวนคำตอบ
- **record_index**: ลำดับของ record

## 🔧 การตั้งค่า

### 1. Environment Variables
ตั้งค่า environment variables หรือ Streamlit secrets:

```bash
export PINECONE_API_KEY="your_api_key"
export PINECONE_ENVIRONMENT_CLOUD="aws"
export PINECONE_ENVIRONMENT_REGION="us-east-1"
export PINECONE_INDEX_NAME="agnos-forums"
```

### 2. ไฟล์ Input
วางไฟล์ forum data (`.txt`) ในโฟลเดอร์ `data_scraped_txt/`

### 3. การตั้งค่า Batch Processing
```python
# Configuration ใน pinecone_batch_upsert.py
INPUT_DIR = "data_scraped_txt"
OUTPUT_DIR = "upserted_data"
BATCH_SIZE = 20  # Process documents in batches
```

## 📈 ผลลัพธ์

### 1. Pinecone Index
- ข้อมูลจะถูก upsert ขึ้น Pinecone ด้วย embedding dimension 1024
- ใช้ TextEmbedder จาก `utils.py` (BGE-M3 model)
- แบ่ง chunk ขนาด 1000 ตัวอักษร พร้อม overlap 150 ตัวอักษร
- Content ถูกเก็บใน metadata เพื่อการ retrieve

### 2. Summary Files
- `upserted_data/batch_processing_summary.json` - สรุปการประมวลผลแบบละเอียด

### 3. Console Output
แสดงความคืบหน้าและสถิติการประมวลผล:
```
📊 Total records extracted: 150
📈 Final Statistics:
   • Total records extracted: 150
   • Records processed: 120
   • Records skipped: 25
   • Records with errors: 5
   • Documents upserted: 240
   • Batches processed: 12
   • Unique sources: 120
```

## 🎯 ตัวอย่างข้อมูลที่ประมวลผล

จากข้อมูลตัวอย่าง:
```
สวัสดีค่ะ อยากจะปรึกษาค่ะในช่วง2-3วันที่ผ่านมามีความรู้สึกดาวน์มากๆ... ![thumbs-up](https://www.agnoshealth.com/assets/icons/thumbs-up/thumbs-up-filled-blue.svg)5![thumbs-up](https://www.agnoshealth.com/assets/icons/forum/comment-checked-filled-blue.svg)แพทย์ตอบคำปรึกษาแล้ว](https://www.agnoshealth.com/forums/เสี่ยงต่อโรคซึมเศร้า/2675)![thumbs-up](https://www.agnoshealth.com/assets/icons/forum/share-outline-blue.svg)แชร์กระทู้
```

จะถูกประมวลผลเป็น:
- **source**: `https://www.agnoshealth.com/forums/เสี่ยงต่อโรคซึมเศร้า/2675`
- **forum_posted_timestamp**: "8/22/2022" (สกัดจาก URL หรือเนื้อหา)
- **forum_reply_count**: 3 (นับจาก reply indicators)
- **content**: "สวัสดีค่ะ อยากจะปรึกษาค่ะในช่วง2-3วันที่ผ่านมามีความรู้สึกดาวน์มากๆ..."
- **record_index**: 3 (ลำดับของ record)

## ⚠️ หมายเหตุ

1. **การแบ่ง Record**: ใช้ `\n\n` (double newline) เป็นตัวแบ่ง
2. **การกรองข้อมูล**: ข้ามส่วนที่มีคำน้อยกว่า 20 คำ
3. **การสกัด Metadata**: ใช้ regex patterns เพื่อสกัด URL, วันที่, และ reply count
4. **Smart Deduplication**: ป้องกันการ upsert ข้อมูลซ้ำโดยใช้ source URL
5. **Content Storage**: เก็บ content ใน metadata เพื่อการ retrieve ใน RAG
6. **Error Handling**: มีการจัดการ error และ logging ครบถ้วน
7. **Batch Processing**: ประมวลผลแบบ batch เพื่อประสิทธิภาพและความเสถียร

## 🔍 การตรวจสอบผลลัพธ์

หลังจากรันสคริปต์แล้ว สามารถตรวจสอบได้จาก:

### 1. Console Output
แสดงความคืบหน้าและสถิติการประมวลผลแบบ real-time

### 2. Summary File
ตรวจสอบไฟล์ `upserted_data/batch_processing_summary.json` เพื่อดู:
- จำนวน records ที่ประมวลผล
- จำนวน documents ที่ upsert สำเร็จ
- รายการ unique sources
- ตัวอย่างข้อมูลที่ประมวลผล

### 3. Pinecone Dashboard
ตรวจสอบข้อมูลที่ upsert แล้วใน Pinecone console

### 4. RAG Testing
ทดสอบ RAG ในแอป Streamlit เพื่อดูว่า content ถูก retrieve ได้หรือไม่

## 🚀 การใช้งานกับ RAG

หลังจาก upsert ข้อมูลแล้ว สามารถใช้ RAG ในแอป Streamlit ได้:

1. **รันแอป**: `streamlit run app.py`
2. **เปิด RAG**: เปิดใช้งาน RAG ใน sidebar
3. **ทดสอบ**: ถามคำถามเกี่ยวกับอาการต่างๆ
4. **ตรวจสอบ**: ดู context ที่ถูก retrieve มาจาก forum posts 