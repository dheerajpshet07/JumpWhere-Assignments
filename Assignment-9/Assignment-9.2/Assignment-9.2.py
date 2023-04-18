# Write a txt file which has a word in each line like:
# Hands
# Legs
# India
# Crow
# Rain
# ...

# Write a python code to read the file and store the words in the list

# Write a function to guess a word randomly from the list.

# Now, write a function which asks user to guess the chosen word letter by letter.
# Show "incorrect" message to the wrong guessed letter. 
# Display  letters in the clue word that were guessed correctly. 

# Let say word is EVAPORATE

# >>> Welcome to Hangman!
# _ _ _ _ _ _ _ _ _
# >>> Guess your letter: S
# Incorrect!
# You left with 5 chances to guess.

# >>> Guess your letter: E
# E _ _ _ _ _ _ _ E
# ...
# And so on.


# 1)Only let the user guess 6 times, and tell the user how many guesses they have left.
# Keep track of the letters the user guessed.
# 2) If the user guesses a letter they already guessed, don’t penalize them - let them guess again.
# 3)When the player wins or loses, let them start a new game.

import random

class Hangman:
    def __init__(self):
        self.guess_taken = 0
        self.current_word = ""
        self.guess_word = ""

        with open("data.txt",'r') as file:
            lines = file.readlines()
            self.words = [line.strip() for line in lines]

    def update_guessword(self,i,ch):
        new_str = self.guess_word[:i] +ch+ self.guess_word[i+1:]
        self.guess_word = new_str


    def play(self):
        random_index = random.randint(0,len(self.words)-1)
        self.current_word = self.words[random_index]
        for i in range(0,len(self.current_word)):
            self.guess_word += "_"

        print("\n It is a ",len(self.current_word)," letter word\n")

        while self.guess_taken < 6:
            guess_char = input("Guess a letter:\n")
            self.guess_taken += 1

            if guess_char in self.guess_word:
                print("You have already guessed that letter, try different letter!!\n Don't worry we will not include this guess\n")
                self.guess_taken -= 1
                continue

            if guess_char in self.current_word:
                for i in range(len(self.current_word)):
                    if guess_char == self.current_word[i]:
                        self.update_guessword(i,guess_char)
                print("\n",self.guess_word,"\n")
            else:
                print("Incorrect\n")

            if(self.guess_word == self.current_word):
                print("\n\tCongratulations!! You have guessed it correctly in ",self.guess_taken, " guesses\n\n")
                return

            if (6-(self.guess_taken) == 0):
                print("\n\tSorry!! Your chances are over!! You Lost the game\n\n")
                return

            print("You have ",(6-(self.guess_taken))," guess left")
                    



        

play_status = 1
print("\n\n\t\t Welcome to Hangman \n\n")
print("Rules :\n1)You will be having a 6 chances within that you have to guess that word\n2)If you guessed the correct letter it will be displayed along with their correct placement\n3)If you re-enter already guessed letter number of chances are not deducted\n\n")
while play_status == 1:
    obj = Hangman()
    obj.play()
    play_status = int(input("\n-> Press 1 to play one more time\n->Press 0 to quit\n"))
