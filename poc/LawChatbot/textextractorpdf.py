# -*- coding: utf-8 -*-


#*********************PDF to Text 코드 **********************
import re
import PyPDF2 #PDF에서 text를 추출하기 위해 PyPDF2 외부 라이브러리 사용.
import os

#PyPDF2를 사용하여 PDF에서 text를 뽑아내는 함수
def extract_text_from_pdf(filepath):
  #PDF에서 텍스트 데이터를 모두 추출하는 기능
  path = filepath+'.pdf'
  with open(path, 'rb') as file:
      reader = PyPDF2.PdfReader(file)
      text = ''
      # 모든 페이지에서 텍스트 추출
      for page in reader.pages:
          text += page.extract_text() + '\n'
  filename = os.path.basename(filepath)
  f = open(filename+'_extract'+'.txt', 'w')
  f.write(text)
  f.close

#추출해낸 text에서 필요한 부분만 Parsing하는 함수
def removeNoiseText(filename, remove_extractFile):
  #추출한 텍스트 데이터에서 필요없는 부분 제거하는 기능
  start_sign = 0
  end_sign = 0

  fr = open(filename+'_extract'+'.txt', 'r')
  lines = fr.readlines()
  fr.close
  if remove_extractFile == 1:
    os.remove(filename+'_extract'+'.txt')

  fw = open(filename+'.txt', 'w')
  for line in lines:
    if '총칙' in line:
      start_sign = 1
    if '부칙' in line:
      end_sign = 1
    if '법제처' in line or end_sign == 1 or start_sign != 1:
      continue
    fw.write(line)
  fw.close

def divideContent(node):
    new_list = []
    markers = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '⑪', '⑫', '⑬', '⑭', '⑮', '⑯', '⑰', '⑱', '⑲', '⑳']
    extracted_text = {marker: '' for marker in markers}

    pattern = '|'.join(re.escape(marker) for marker in markers)
    sections = re.split(f'({pattern})', node[len(node)-1])  # 마지막 항목을 분리
    if len(sections) == 1: #marker가 없는 내용일 경우 그대로 return
      new_list.append(node)
      return new_list

    current_marker = None
    for section in sections:
        if section in markers:
            current_marker = section
        elif current_marker:
            extracted_text[current_marker] += section.strip()

    for marker, content in extracted_text.items():
      if content:
        content = content.replace('\n', ' ')  # \n을 공백으로 대체
        # node의 앞부분과 마커+내용을 결합
        tmp = node[:11] + [marker + content]
        new_list.append(tmp)

    return new_list

def makeTable(filename):
  f = open(filename+'.txt','r')
  lines = f.readlines()

  #장에 대한 패턴
  chap_pattern = r'제(\d+)장\s(.+)'
  #절에 대한 패턴
  sec_pattern = r'제(\d+)절\s(.+)'
  #조에 대한 패턴
  art_pattern = r"제(\d+)조\((.*?)\)\s+(.*)"

  chap_num = ""
  chap_name = ""
  sec_num = ""
  sec_name = ""
  art_num = ""
  art_name = ""
  art_content = str("")

  data = []
  content_sign = 0

  for line in lines:
    if "물질안전보건자료" in line:
      line = line.replace("물질안전보건자료","물질안전보건자료(MSDS)")
    chap_match = re.findall(chap_pattern, line)
    if chap_match:
      if content_sign == 1:
        tmp = [filename, None, None, chap_num, chap_name, sec_num, sec_name, None, None, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
        content_sign = 0
      chap_num = '제'+chap_match[0][0]+'장'
      chap_name = chap_match[0][1]
      sec_num, sec_name, art_num, art_name, art_content = None, None, None, None, None
      continue
    sec_match = re.findall(sec_pattern, line)
    if sec_match:
      if content_sign == 1:
        tmp = [filename, None, None, chap_num, chap_name, sec_num, sec_name, None, None, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
        content_sign = 0
      sec_num = '제'+sec_match[0][0]+'절'
      sec_name = sec_match[0][1]
      art_num, art_name, art_content = None, None, None
      continue
    art_match = re.findall(art_pattern, line)
    if art_match:
      if content_sign == 1:
        tmp = [filename, None, None, chap_num, chap_name, sec_num, sec_name, None, None, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
      art_num = '제'+art_match[0][0]+'조'
      art_name = '('+art_match[0][1]+')'
      art_content = art_match[0][2]
      content_sign = 1
      continue
    art_content = str(art_content)+line

  #마지막 조항까지 추가하기 위함.
  tmp = [filename, None, None, chap_num, chap_name, sec_num, sec_name, None, None, art_num, art_name, art_content]
  nodes = divideContent(tmp)
  for node in nodes:
    data.append(node)
  #data.append(tmp)
  return data


#편, 장, 절, 관, 조로 구성되어 있는 법령 텍스트를 List형태로 추출해내는 함수(DB에 삽입하기 위함)
def make_RegulationTable(filename):
  f = open(filename + '.txt','r')
  lines = f.readlines()

  #편에 대한 패턴
  part_pattern = r'제(\d+)편\s(.+)'
  part_num = ""
  part_name = ""
  #장에 대한 패턴
  chap_pattern = r'제(\d+)장\s(.+)'
  chap_num = ""
  chap_name = ""
  #절에 대한 패턴
  sec_pattern = r'제(\d+)절\s(.+)'
  sec_num = ""
  sec_name = ""
  #관에 대한 패턴
  par_pattern = r'제(\d+)관\s(.+)'
  par_num = ""
  par_name = ""
  #조에 대한 패턴
  art_pattern = r"제(\d+)조\((.*?)\)\s+(.*)"
  art_num = ""
  art_name = ""
  art_content = str("") #조항 내용을 끝까지 이어붙이기 위해서는 초기 값이 string일 필요가 있음.

  data = []
  content_sign = 0

  for line in lines:
    part_match = re.findall(part_pattern, line)
    if part_match:
      if content_sign == 1:
        tmp = [filename, part_num, part_name, chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
        content_sign = 0
      part_num = '제'+part_match[0][0]+'편'
      part_name = part_match[0][1]
      chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content = None, None, None, None, None, None, None, None, None
      continue
    chap_match = re.findall(chap_pattern, line)
    if chap_match:
      if content_sign == 1:
        tmp = [filename, part_num, part_name, chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
        content_sign = 0
      chap_num = '제'+chap_match[0][0]+'장'
      chap_name = chap_match[0][1]
      sec_num, sec_name, par_num, par_name, art_num, art_name, art_content = None, None, None, None, None, None, None
      continue
    sec_match = re.findall(sec_pattern, line)
    if sec_match:
      if content_sign == 1:
        tmp = [filename, part_num, part_name, chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
        content_sign = 0
      sec_num = '제'+sec_match[0][0]+'절'
      sec_name = sec_match[0][1]
      par_num, par_name, art_num, art_name, art_content = None, None, None, None, None
      continue
    par_match = re.findall(par_pattern, line)
    if par_match:
      if content_sign == 1:
        tmp = [filename, part_num, part_name, chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
        content_sign = 0
      par_num = '제'+par_match[0][0]+'관'
      par_name = par_match[0][1]
      art_num, art_name, art_content = None, None, None
      continue
    art_match = re.findall(art_pattern, line)
    if art_match:
      if content_sign == 1:
        tmp = [filename, part_num, part_name, chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content]
        nodes = divideContent(tmp)
        for node in nodes:
          data.append(node)
        #data.append(tmp)
      art_num = '제'+art_match[0][0]+'조'
      art_name = '('+art_match[0][1]+')'
      art_content = art_match[0][2]
      content_sign = 1
      continue
    art_content = str(art_content)+line

  #마지막 조항까지 추가하기 위함.
  tmp = [filename, part_num, part_name, chap_num, chap_name, sec_num, sec_name, par_num, par_name, art_num, art_name, art_content]
  nodes = divideContent(tmp)
  for node in nodes:
    data.append(node)
  #data.append(tmp)
  return data



#Main Function
path = './Data/Puredata/RealPure/' #현재 환경 Path에 맞게 설정
file = ['안전보건규칙','산업안전보건법', '산업안전보건법시행규칙','산업안전보건법시행령', '중대재해처벌법']
expection = '안전보건규칙' #기존 법령과 구조가 다른 법령 구분 필요.
listForDB = []

for filename in file :
  extract_text_from_pdf(path+filename)
  removeNoiseText(filename, remove_extractFile=1) #만약 기존에 추줄된 원본 텍스트파일을 삭제하고 싶다면 1, 아니면 0
  if filename == expection:
    listForDB.append(make_RegulationTable(filename))
  else:
    listForDB.append(makeTable(filename))
#************************PostgreSQL DB에 Insert하는 코드 *********************************8

import psycopg2 #PostgreSQL DB와 연결하기 위해 psycopg2 외부 라이브러리 사용
try:
    #table_name = {'중대재해처벌법':'DisasterAct', '산업안전보건법시행령':'EnforDecree', '산업안전보건법시행규칙':'EnforRule', '산업안전보건법':'IndusAct', '안전보건규칙':'Regulation'}
    table_name = 'each_paragraph'
    conn = psycopg2.connect(host='localhost', dbname='law', user='user1', password='1234', port=5432)
    cursor = conn.cursor()
    for current in listForDB:
      args_str = ", ".join([cursor.mogrify('(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', row).decode('utf-8') for row in current])
      sql = "INSERT INTO public.{table}(name,part_num, part_name, chap_num, char_name, sec_num, sec_name, para_num, para_name, art_num, art_name, content) VALUES {data};".format(data=args_str,table=table_name)
      try:
          cursor.execute(sql)
      except Exception as e:
          print("Insert Error: ", e)
      conn.commit()

except Exception as e:
    print(f"데이터베이스 연결에 실패했습니다: {e}")
finally:
    # 데이터베이스 연결 종료
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL 연결이 종료되었습니다.")
