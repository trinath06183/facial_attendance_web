from django.contrib import admin
from .models import UserProfile, Student, Attendance


# Register your models here.


admin.site.register(UserProfile)
admin.site.register(Student)
admin.site.register(Attendance)