from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Voter, Vote
from .face_recognition import register_face, recognize_face
import datetime
import csv
import os
import pyttsx3
from django.core.exceptions import ObjectDoesNotExist

def speak(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def check_if_voted(aadhar_number):
    try:
        if os.path.exists("Votes.csv"):
            with open("Votes.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0] == aadhar_number:
                        return True
        return False
    except Exception as e:
        print(f"Error checking vote status: {str(e)}")
        return False

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == "POST":
        try:
            aadhar_number = request.POST['aadhar_number']
            if not aadhar_number:
                return render(request, 'register.html', {'error': 'Aadhar number is required'})

            if register_face(aadhar_number):
                try:
                    Voter.objects.create(aadhar_number=aadhar_number)
                    speak("Registration successful")
                    return redirect('home')
                except Exception as e:
                    print(f"Database error: {str(e)}")
                    return render(request, 'register.html', {'error': 'Error creating voter in database'})
            else:
                return render(request, 'register.html', {'error': 'Face registration failed. Please try again.'})
        except Exception as e:
            print(f"Registration error: {str(e)}")
            return render(request, 'register.html', {'error': 'An error occurred during registration'})
    
    return render(request, 'register.html')

def vote(request):
    if request.method == "POST":
        try:
            # Authenticate voter using face recognition
            aadhar_number = recognize_face()

            if aadhar_number is None:
                speak("You are not registered")
                return render(request, 'not_registered.html', {'message': 'You are not registered.'})

            # Check if the face data exists in the "data" folder
            face_data_file = os.path.join('data', f'{aadhar_number}.pkl')
            if not os.path.exists(face_data_file):
                speak("Face data not found")
                return render(request, 'not_registered.html', {'message': 'Face data not found.'})

            if check_if_voted(aadhar_number):
                speak("You have already voted")
                return render(request, 'already_voted.html')

            candidate = request.POST.get('candidate')
            if not candidate:
                speak("Please select a candidate")
                return render(request, 'vote.html', {'error': 'Please select a candidate'})

            date = datetime.date.today()
            time = datetime.datetime.now().strftime("%H:%M:%S")

            try:
                voter = Voter.objects.get(aadhar_number=aadhar_number)
                Vote.objects.create(voter=voter, candidate=candidate)

                # Save vote to CSV
                with open("Votes.csv", "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([aadhar_number, candidate, date, time])

                speak("Thank you for voting")
                return render(request, 'success.html')
            
            except ObjectDoesNotExist:
                speak("Voter not found in database")
                return render(request, 'not_registered.html', {'message': 'Voter not found in database.'})
            except Exception as e:
                print(f"Error saving vote: {str(e)}")
                speak("Error saving your vote")
                return render(request, 'vote.html', {'error': 'Error saving your vote. Please try again.'})

        except Exception as e:
            print(f"Voting error: {str(e)}")
            speak("An error occurred")
            return render(request, 'vote.html', {'error': 'An error occurred. Please try again.'})

    return render(request, 'vote.html')