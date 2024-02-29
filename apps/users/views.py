import re
from django import http
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
from django.contrib.auth import get_user_model

User = get_user_model()


# Create your views here.

class RegisterView(View):
    def get(self, request):
        return render(request, 'register.html')

    def post(self, request):
        data = request.POST
        username = data.get('username')
        password = data.get('password')
        password2 = data.get('password2')
        mobile = data.get('mobile')
        allow = data.get('allow')
        if not all([username, password, password2, mobile, allow]):
            return http.HttpResponseBadRequest('缺少輸入參數')
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_-]{5,20}$', username):
            return http.HttpResponseBadRequest('用戶名不符合規則')
        if not re.match(r'^[0-9A-Za-z]{8,20}$', password):
            return http.HttpResponseBadRequest('密碼不符合規則')
        if password != password2:
            return http.HttpResponseBadRequest('密碼不一致')
        if not re.match(r'^09\d{8}$', mobile):
            return http.HttpResponseBadRequest('手機號碼不符合規則')
        try:
            user = User.objects.create_user(username=username, password=password, mobile=mobile)
        except Exception as e:
            # return render(request, 'register.html', context={'error_message': '數據庫異常'})
            return http.HttpResponse('資料庫錯誤')

        from django.contrib.auth import login
        login(request, user)
        return redirect(reverse('contents:index'))
        # return http.HttpResponse('註冊成功!')


class UsernameCountView(View):
    def get(self, request, username):
        try:
            count_name = User.objects.filter(username=username).count()
        except Exception as e:
            return http.JsonResponse({'code': 400, 'errmsg': '數據庫異常'})
        return http.JsonResponse({'code': 0, 'count': count_name})


class UserphoneCountView(View):
    def get(self, request, phone):
        try:
            count_phone = User.objects.filter(mobile=phone).count()
        except Exception as e:
            return http.JsonResponse({'code': 400, 'errmsg': '數據庫異常'})
        return http.JsonResponse({'code': 0, 'count_phone': count_phone})


class LoginView(View):
    def get(self, request):
        return render(request, 'login.html')

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')
        remembered = request.POST.get('remembered')
        if not all([username, password]):
            return http.HttpResponseBadRequest('缺少必要參數')
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_-]{5,20}$', username):
            return http.HttpResponseBadRequest('用戶名不符合規則')
        if not re.match(r'^[0-9A-Za-z]{8,20}$', password):
            return http.HttpResponseBadRequest('密碼不符合規則')

        from django.contrib.auth import authenticate
        from django.contrib.auth import login
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            if remembered == 'on':
                request.session.set_expiry(30 * 24 * 3600)
            else:
                request.session.set_expiry(0)
            return redirect(reverse('contents:index'))
        else:
            return render(request, 'login.html', context={'account_errmsg': '帳號或密碼錯誤'})
