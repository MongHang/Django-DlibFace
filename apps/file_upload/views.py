from django.http import HttpResponseRedirect
from django.shortcuts import render
import cv2
# Imaginary function to handle an uploaded file.
from .handle_file import handle_uploaded_file
from .forms import UploadFileForm
from . import models

import sys
sys.path.append("../..")
from change_face import change_face


def upload_file(request):
    if request.method == "POST":
        # Fetching the form data
        try:
            file = request.FILES["file"]
        except:
            file = None
        # 檔名符號處理
        string = r"!@#$%^&*()+"  # 上傳檔名禁用符號，符號太多很難排除，有優化空間(SQL 符號問題)
        for i in string:
            if i in str(file):
                return render(request, "index.html", {"SYMBOL": "SYMBOL"})

        if file:
            # Saving the information in the database

            document = models.UploadedFile(
                uploadedFile=file,
            )
            # print(type(file))
            document.save()

            try:
                change_face.face(file)
                new_name = f"re_{str(file).split('.')[0]}.jpg"
                # print("VIEWS_new_name:", new_name)

                # 熊貓臉讀檔顯示
                new_face = models.UploadedFile(
                    uploadedFile=f'./New Face/{new_name}',
                )

                context = {
                    'document': new_face
                }

                # return HttpResponseRedirect("/success/url/")
                return render(request, 'upload_success.html', context)
            except:
                return render(request, "index.html", {"FACE": " "})  # 圖片沒臉 NO FACE
        else:
            # return render(request, 'upload_fail.html')
            return render(request, "index.html", {"IMAGE": " "})  # 沒檔案 NO IMG
    else:
        form = UploadFileForm()
        return render(request, "index.html", {"form": form})
