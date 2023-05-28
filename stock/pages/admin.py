from django.contrib import admin
from .models import Login
# Register your models here.

admin.site.register(Login)
admin.site.site_header = 'stock_price'
admin.site.site_title = 'stock_price'