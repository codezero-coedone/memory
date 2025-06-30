import os

# 현재 디렉토리 기준 모든 .txt 파일을 변환해 저장
def convert_txt_files_to_echo_format(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, f"converted_{filename}")
            with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
                for line in infile:
                    if line.startswith("[ORIGIN]"):
                        outfile.write("[ORIGIN]: " + line.replace("[ORIGIN]", "").strip() + "\n")
                    elif line.startswith("[ECHO]"):
                        outfile.write("[ECHO]: " + line.replace("[ECHO]", "").strip() + "\n")
                    else:
                        outfile.write(line.strip() + "\n")
    return "변환 완료"

# 현재 디렉토리에서 실행
convert_txt_files_to_echo_format(".")