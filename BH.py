import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# إعداد الواجهة بشكل احترافي
st.set_page_config(page_title="مترجم اللهجة البحرينية", page_icon="🇧🇭")
st.title("🇧🇭 نظام الترجمة الآلية للهجة البحرينية")

# تحديد المسار (تأكدي أن المجلد bahraini_ai بجانب هذا الملف)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bahraini_ai")

@st.cache_resource
def load_model_components():
    if os.path.exists(model_path):
        try:
            # استخدام use_fast=False يحل مشكلة الـ NoneType مع موديلات Helsinki
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            return tokenizer, model
        except Exception as e:
            st.error(f"فشل تحميل الملفات من المجلد: {e}")
            return None, None
    else:
        st.error(f"المجلد '{model_path}' غير موجود!")
        return None, None

tokenizer, model = load_model_components()

# واجهة المستخدم
user_text = st.text_input("فصحى:", placeholder="مثال: أين تذهب؟")

if st.button("تحويل إلى البحريني"):
    if user_text and tokenizer and model:
        with st.spinner("جاري المعالجة..."):
            try:
                # 1. Tokenization (الترميز)
                # أضفنا الكود الخاص باللغة العربية يدوياً لحل مشكلة الـ replace
                input_ids = tokenizer.encode(user_text, return_tensors="pt")
                
                # 2. Generation (الاستنتاج)
                outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
                
                # 3. Decoding (فك الترميز)
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.success(f"النتيجة: {prediction}")
            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة التقنية: {e}")
    elif not user_text:
        st.warning("الرجاء إدخال نص أولاً")

st.info("ملاحظة: هذا النموذج تم تدريبه باستخدام الـ Fine-tuning على مجموعة بيانات بحرينية.")