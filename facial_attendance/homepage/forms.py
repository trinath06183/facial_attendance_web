from django import forms
from django.contrib.auth.models import User
from .models import Student, UserProfile

class StudentRegistrationForm(forms.ModelForm):
    name = forms.CharField(max_length=100, label="Full Name")
    roll_number = forms.CharField(max_length=20, label="Roll Number")
    department = forms.CharField(max_length=100, label="Department")
    password = forms.CharField(widget=forms.PasswordInput(), label="Choose Password")
    confirm_password = forms.CharField(widget=forms.PasswordInput(), label="Confirm Password")

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match")
        return cleaned_data
