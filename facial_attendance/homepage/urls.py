from django.urls import path
from . import views
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_page, name='login'),
    path('dashboard/', views.dashboard_redirect, name='dashboard'),
    path('dashboard/admin/', views.admin_dashboard, name='admin_dashboard'),
    path('dashboard/teacher/', views.teacher_dashboard, name='teacher_dashboard'),
    path('dashboard/student/', views.student_dashboard, name='student_dashboard'),
    path('dashboard/student/profile/', views.student_profile, name='student_profile'),
    path('logout/', views.logout_view, name='logout'),
    
    path('register-student/', views.register_student, name='register_student'),
    path('register-teacher/', views.register_teacher, name='register_teacher'),
    path('enroll-face/', views.enroll_face, name='enroll_face'),
    
    path('all-students/', views.view_all_students, name='view_all_students'),
    path('manage-faculty/', views.manage_faculty, name='manage_faculty'),
    path('settings/', views.manage_settings, name='manage_settings'),
    path('today-attendance/', views.todays_attendance, name='todays_attendance'),
    path('manual-entry/', views.manual_attendance, name='manual_attendance'),
    path('remove-student/<int:student_id>/', views.remove_student, name='remove_student'),
    path('remove-teacher/<int:teacher_id>/', views.remove_teacher, name='remove_teacher'),
    path('edit-student/<int:student_id>/', views.edit_student, name='edit_student'),
    path('edit-teacher/<int:teacher_id>/', views.edit_teacher, name='edit_teacher'),
    path('edit-attendance/<int:record_id>/', views.edit_attendance, name='edit_attendance'),
    
    path('export-reports/', views.export_reports_page, name='export_reports_page'),
    path('export-csv/', views.export_attendance_csv, name='export_csv'),
    path('export-pdf/', views.export_attendance_pdf, name='export_pdf'),
    path('student-export-pdf/', views.student_export_pdf, name='student_export_pdf'),
    
    path('face-scan/', views.face_scan_page, name='face_scan'),
    path('face-scan-login/', views.face_scan_login_view, name='face_scan_login'),
    path('process-client-frame/', views.process_client_frame, name='process_client_frame'),
    path('capture-enrollment-frame/', views.capture_enrollment_frame_api, name='capture_enrollment_frame'),
    path('stop-camera/', views.stop_camera, name='stop_camera'),
    path('verify-teacher-face/', views.verify_teacher_face, name='verify_teacher_face'),
    path('capture-progress/', views.capture_progress_api, name='capture_progress'),
    path('save-face-data/', views.save_face_data, name='save_face_data'),
    path('reset-session/', views.reset_recognition_session, name='reset_session'),
    path('check-face-status/', views.check_face_status, name='check_face_status'),
]