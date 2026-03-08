
PS D:\4th_sem_project\implementation\facial_attendance_10> py -m venv venv
#        creating vertual env



PS D:\4th_sem_project\implementation\facial_attendance_10> .\venv\Scripts\activate
#        activating vertual env



(venv) PS D:\4th_sem_project\implementation\facial_attendance_10> pip install django 
#        installing django in venv


#########################################################################################


(venv) PS D:\4th_sem_project\implementation\facial_attendance_10> django-admin startproject faceial_attenfance
#       creating the project



(venv) PS D:\4th_sem_project\implementation\facial_attendance_10> cd .\faceial_attenfance\
#       go to project dir


(venv) PS D:\4th_sem_project\implementation\facial_attendance_10\faceial_attenfance> py .\manage.py startapp homepage 
#       creating the app/funtions


#########################################################################################

#         creating a template folder and create the html file
#         Connect the View and URL
#         Tell Django how to find and serve your HTML page. 
#       Update the View (homepage/views.py)
#       def home(request):
#             return render(request, 'homepage/index.html')
<!-- 
create the file
Create App URLs (homepage/urls.py):

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
] 
-->

<!-- 
edit in Project URLs (mysite/urls.py):

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('homepage.urls')),
]
 -->

 #      runthe server
 #  (venv) PS D:\4th_sem_project\implementation\facial_attendance_10\faceial_attenfance> py .\manage.py runserver

# CHANGE THIS LINE INSIDE SETTINGS.PY - TEMPLATES 
<!--     'DIRS': [BASE_DIR / 'templates'],      -->

#########################################################################################

#       now implementing login page and linked to home page

#       1st add """path('login/', views.login_page, name='login'),""" in homepahe/urls.py 
#       2ND add """def login_page(request):
#                       return render(request, 'login.html')""" in homepahe/views.py 

#       
#      now create the login.html file in templates/login.html 
#       
#       
#       
#       
#       
#       
#       
#########################################################################################
#       now implement backend logics     
#       
#       install required Libraries 1st
#       
#  1.        pip install opencv-python numpy
#       
#  2.        Create the Camera Logic (camera.py)
#       Inside your homepage app folder, create a new file called camera.py. This keeps your video processing code separate from your views.
#  3.       Update views.py
#        Now, create a "Stream" view that pushes these frames to the browser.         
#  4.         Map the URLs
#       Update your homepage/urls.py so the browser knows where to find the camera stream. 
#  5.       Create the UI (face_scan.html)
#       This is the page that shows the video. We will make it look like a high-tech scanner.     
#  6.       Link the "Login with Face" Button
#       Finally, go back to your login.html and update the "Login with Face Scan" button to point to this new page:

#       In login.html:     
#  7.            
#       
#########################################################################################
#       
#       now implement the logic and funcinality for login
#       
#       
##########################################################################################
Step 1: 
    Update your Database Model (models.py)
    To differentiate between a Student and a Teacher, we will create a UserProfile model that links to the standard Django User.

    Run these commands in your terminal to update the database:

``````````Bash
python manage.py makemigrations
python manage.py migrate
``````````
Step 2: 
    Create a Role-Based Redirection (views.py)
    When a user logs in (either via password or face scan), Django needs to know where to send them.

    Update your views.py to include a "Dashboard Redirector":

Step 3: 
    Create the Dashboard Templates
    Organize your templates folder like this for a clean project structure:

    templates/dashboards/student_dash.html

    templates/dashboards/teacher_dash.html

    templates/dashboards/admin_dash.html

Step 4: 
    Update urls.py
    Link the dashboard redirector so it works after login.

Step 5: 
    Register Roles in Admin
    To test this immediately, you need to be able to assign roles to yourself in the Django Admin panel. Open homepage/admin.py:


CREATING SUPER USER/ADMIN
    How to implement this in your system now:
    Create a Superuser: Run python manage.py createsuperuser in your terminal.
    username for superuser = admin
    password for superuser = admin123
    Go to Admin: Log in at http://127.0.0.1:8000/admin.

    Assign Role: You will see UserProfiles. Create one for your user and set the role to Admin or Teacher.
    user= t1
    pass= t106183@

    Test: Go back to your login page, sign in, and you will see the specific dashboard for that role.



























