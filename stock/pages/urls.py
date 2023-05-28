from django.urls import path
from . import views

urlpatterns =[
    
    path('', views.Home , name = 'Home'),
    path('Amazon', views.Amazon , name = 'Amazon'),
    path('Apple', views.Apple , name = 'Apple'),
    path('Intel', views.Intel , name = 'Intel'),
    path('Google', views.Google , name = 'Google'),
    path('Nvidia', views.Nvidia , name = 'Nvidia'),
    path('Meta', views.Meta , name = 'Meta'),
    path('Microsoft', views.Microsoft , name = 'Microsoft'),
    path('Netflix', views.Netflix , name = 'Netflix'),  
    path('Tesla', views.Tesla , name = 'Tesla'), 
    path('signup', views.signup , name ='signup'),
    path('signin', views.signin , name ='signin'),
    path('signout', views.signout , name ='signout'),
    path('Contact', views.Contact , name = 'Contact'),


    path('AboutApple', views.AboutApple , name = 'AboutApple'),
    path('PredictionApple', views.PredictionApple , name = 'PredictionApple'),
    
    path('AboutAmazon', views.AboutAmazon , name = 'AboutAmazon'),
    path('PredictionAmazon', views.PredictionAmazon , name = 'PredictionAmazon'),
   
    path('AboutIntel/', views.AboutIntel, name='AboutIntel'),
    path('PredictionIntel', views.PredictionIntel , name = 'PredictionIntel'),
    
    path('AboutGoogle', views.AboutGoogle , name = 'AboutGoogle'),
    path('PredictionGoogle', views.PredictionGoogle , name = 'PredictionGoogle'),
    
    path('AboutNvidia', views.AboutNvidia , name = 'AboutNvidia'),
    path('PredictionNvidia', views.PredictionNvidia , name = 'PredictionNvidia'),
    
    path('AboutMeta', views.AboutMeta , name = 'AboutMeta'),
    path('PredictionMeta', views.PredictionMeta , name = 'PredictionMeta'),
   
    path('AboutMicrosoft', views.AboutMicrosoft , name = 'AboutMicrosoft'),
    path('PredictionMicrosoft', views.PredictionMicrosoft , name = 'PredictionMicrosoft'),
   
    path('AboutNetflix', views.AboutNetflix , name = 'AboutNetflix'),
    path('PredictionNetflix', views.PredictionNetflix , name = 'PredictionNetflix'),
   
    path('AboutTesla', views.AboutTesla , name = 'AboutTesla'),
    path('PredictionTesla', views.PredictionTesla , name = 'PredictionTesla'),
    


]  