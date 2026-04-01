import asyncio
import os
import streamlit as st
from pipeline_core import run_pipeline

st.set_page_config(page_title='Аналіз наративів', layout='wide')

st.markdown("""
<style>
.results-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.results-table th { background: #f0f2f6; padding: 8px 12px; text-align: left; border: 1px solid #ddd; white-space: nowrap; }
.results-table td { padding: 8px 12px; border: 1px solid #ddd; vertical-align: top; word-break: break-word; white-space: normal; }
.results-table tr:nth-child(even) { background: #fafafa; }
</style>
""", unsafe_allow_html=True)

st.title('Аналіз економічних наративів')
st.markdown('Вставте текст — пайплайн знайде економічний контент і витягне структуровані фрейми.')

st.markdown('Приклад промови за посиланням: [president.gov.ua](https://www.president.gov.ua/news/virimo-sho-sili-ameriki-dostatno-shob-razom-iz-nami-razom-z-95989)')

text = st.text_area('Текст для аналізу', height=300,
                    placeholder='Вставте сюди текст промови, статті або допису...')

if st.button('Аналізувати', type='primary'):
    if not text.strip():
        st.warning('Вставте текст.')
    else:
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            st.error('Не задано OPENAI_API_KEY')
        else:
            with st.spinner('Аналізую...'):
                try:
                    df = asyncio.run(run_pipeline(text=text, api_key=api_key))
                    if df.empty:
                        st.info('Економічного контенту не знайдено.')
                    else:
                        st.success(f'Готово. Знайдено фреймів: {len(df)}')
                        cols = ['paragraph', 'object', 'problem', 'short_description',
                                'government_actions', 'action_status',
                                'responsible_institutions', 'urgency', 'narrative_tag']
                        st.markdown(df[cols].to_html(classes='results-table', index=False), unsafe_allow_html=True)
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f'Помилка: {e}')
