from django.db import models
from django.contrib.auth.models import User 

class UserProfile(models.Model):
    ROLE_CHOICES = (
        ("student", "Student"),
        ("teacher", "Teacher"),
        ("admin", "Admin"),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default="student")
    # face_id stores the integer User PK for OpenCV matching
    face_id = models.IntegerField(null=True, blank=True)  
    is_face_registered = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.role}"

class Student(models.Model):
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE, limit_choices_to={'role': 'student'})
    roll_number = models.CharField(max_length=20, unique=True)
    department = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.user_profile.user.username} ({self.roll_number})"

class Teacher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='teacher_profile')
    name = models.CharField(max_length=100)
    department = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.name} ({self.department})"

from django.utils import timezone

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    time = models.TimeField(default=timezone.now)
    status = models.CharField(max_length=10, default="Present")

    def __str__(self):
        return f"{self.student.roll_number} - {self.date}"

class AppSetting(models.Model):
    name = models.CharField(max_length=100, unique=True)
    is_enabled = models.BooleanField(default=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        status = "Enabled" if self.is_enabled else "Disabled"
        return f"{self.name} ({status})"