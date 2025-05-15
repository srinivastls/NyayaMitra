from milvus_query import *
search_text = 'Section 437 of H MC Act Award costs'
prayer_result = prayers_search(search_text)
print(prayer_result)
prayer = prayer_result[0] if prayer_result else ''
search_text = f'''section: Section 437 of H MC Act Award costs

Sample prayer:{prayer}
'''
print(judgement_search(search_text))
