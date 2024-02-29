import re
from django.contrib.auth.backends import ModelBackend
from .models import User


class UsernameModelBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            if re.match(r'^09\d{8}$', username):
                user = User.objects.get(mobile=username)
            else:
                user = User.objects.get(username=username)
        except:
            return None

        if user.check_password(password):
            return user
