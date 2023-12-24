from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import os
import io
import csv
import base64
import cv2 as cv
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
from grader import Grader
import matplotlib.pyplot as plt
from pydantic import BaseModel
from pathlib import Path  # Để xử lý đường dẫn file

#uvicorn mycv2:app --reload
app = FastAPI()
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/temp_img", StaticFiles(directory="temp_img"), name="temp_img")
templates = Jinja2Templates(directory="html")
saved_scores = []

# Tạo một lớp Pydantic để chấp nhận dữ liệu tải lên
class ImageUpload(BaseModel):
    file: UploadFile

# Định nghĩa route để hiển thị trang HTML cho việc tải lên ảnh
@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request,):
    sns.set_style("whitegrid", {'axes.grid' : False})
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(saved_scores)<=3:
        plot = sns.kdeplot([0], color='skyblue', ax=ax, fill=True)
        ax.set_title('Chấm Nhiều Hơn 3 Bài Để Hiển Thị Phổ Điểm!')
        ax.set_ylabel('Tần Suất')
        ax.set_xticks(range(1, 11))
    else:
        plot = sns.kdeplot(saved_scores, color='skyblue', ax=ax, fill=True)
        ax.set_title('Phổ Điểm Tất Cả Bài Đã Chấm')
        ax.set_ylabel('Tần Suất')
        ax.set_xticks(range(1, 11))
    plt.savefig('temp_img/score_chart.png', format='png')
    return templates.TemplateResponse('home_page.html', {"request": request})

@app.post("/grade/score", response_class=HTMLResponse)
async def grade_image(request: Request, image: UploadFile, user_csv: UploadFile):
    # Đọc dữ liệu ảnh từ file tải lên
    file_content = await image.read()
    nparr = np.frombuffer(file_content, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    # Tạo dummy để dùng lấy thông tin người làm bài
    dummy = Grader(img=image, test_answers={1:'A'}, max_score=10)
    checkmade = False
    info = dummy.get_info()
    # Kiểm tra mã đề và lấy đáp án theo mã
    csv_contents = await user_csv.read()
    ans_file = io.StringIO(csv_contents.decode("utf-8"))
    ans = pd.read_csv(ans_file, dtype=str, header=None)
    for i in range(len(ans)):
        if ans[0][i] == info[1]: 
            checkmade=True
            break
    if checkmade == True:
        test_answers = ans.T[i][1:].to_dict()
    else:
        raise HTTPException(status_code=404, detail="Mã đề không tồn tại trong file chấm điểm!")
    try:
        # Sử dụng lớp Grader từ cv.py để chấm điểm
        grader = Grader(img=image, test_answers=test_answers, max_score=10)
        ans, info, score, result_img = grader.grade()
        saved_scores.append(score)
        score = f"{score:.2f}"
    
        # Mở hoặc tạo tệp CSV để lưu trữ thông tin
        with open('results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # Điền thông tin vào tệp CSV sau khi chấm xong
            writer.writerow([info[0], info[1], ans, len(ans), score])
        # Chuyển đổi ảnh thành dạng Base64 để hiển thị trong trang HTML
        image_base64 = base64.b64encode(cv.imencode('.png', result_img)[1]).decode()

        return templates.TemplateResponse('chamdiem_page.html', {
            "request": request,
            "info_0": info[0], 
            "info_1": info[1], 
            "cr_ans": len(ans), 
            "score": score, 
            "image_base64": image_base64,
            "ans": ans
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")

@app.get("/download-csv")
async def download_csv():
    # Đảm bảo rằng tệp CSV tồn tại
    if not os.path.exists("results.csv"):
        raise HTTPException(status_code=404, detail="Tệp CSV không tồn tại")
    # Trả về tệp CSV cho tải về
    return FileResponse("results.csv", headers={"Content-Disposition": "attachment; filename=results.csv"})