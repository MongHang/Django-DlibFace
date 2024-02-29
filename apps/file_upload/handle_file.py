def handle_uploaded_file(f):
    with open("./img/photo.jpg", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
