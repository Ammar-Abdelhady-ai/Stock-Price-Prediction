# in forms.py
from django import forms
import joblib


crrency_names = joblib.load(r".\pages\save data\crrency_names.h5")

class MyForm(forms.Form):
    my_list_choices = [(item, item) for item in crrency_names]
    my_list_field = forms.ChoiceField(choices=my_list_choices, label='Choose an Crrency names')
    number_choices = [(num, num) for num in range(1, 8)]
    number_field = forms.ChoiceField(choices=number_choices, label='Choose number of days')

