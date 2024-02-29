from django.urls import path, re_path
from .views import upload_file

urlpatterns = [
    re_path(r'^upload/$', upload_file, name='upload_file'),
]


from django.conf import settings
# 添加用於處理媒體文件的URL路由
if settings.DEBUG:
    from django.conf import settings
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


