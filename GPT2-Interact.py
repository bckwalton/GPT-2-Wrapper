import gpt_2_simple as gpt2
import os
import sys
import shutil
from shutil import copytree
import time
print('')
# '1558M' Is now available but is crippling!

# Validation Loop on Download of weights
# Prompts:
# 'President Obama, after a recent mass shooting incident (in which a man injured more than 20 people and killed 7), met in the rose garden with reporters for a public address. The text that follows is the unmodified transcript of his speech:'
# 'Elon Musk, met with reporters today to discuss the new break through in General AI by OpenAI and what it means for the world moving forward. The following is his uninterrupted transcript: '
x = True
while x:
    response = input('What model should I use for this?: ').lower()
    if response == '117m':
        model_name = "117M"
        x = False
    elif response == '345m':
        model_name = "345M"
        x = False
    elif response == '774m':
        model_name = "774M"
        x = False
    else:
        print('Please respond with either 117M 345M or 774M, there are no other available options.')
fil = 'JOHN_WICK_SCRIPT.txt'
seedTextStart = 'After working on his AI assistant for the past 3 months, Brian finally turned it on, heres their Dialogue: Brian: "Hello?" AI: "Hello sir, how can I be of assistance?"'
seedTextEnd = "Thank you, and good night."
if not os.path.isdir('./models/'+str(model_name)):
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/117M/
    try:
        os.makedirs('checkpoint')
        copytree('./models/'+model_name,'./checkpoint/run'+model_name)
    except FileExistsError:
        print('I tried to switch from the previous model but it seems a model was already used previously, in the pursuit of saving potentialy expensive rounds of retraining Ill use the one thats already present')
sess = gpt2.start_tf_sess()
rounds = 10
x = True
while x:
    sess_flag = False
    response = input('Finetune? (y/n): ').lower()
    if response == 'y':
        sess_flag = True
        if int(model_name[0:-1]) > 345:
            print('Models higher than 345M are simply un-finetuneable on modern GPUs, I will only generate.')
            #shutil.rmtree('./checkpoint/', ignore_errors=False, onerror=None)
            #os.makedirs('checkpoint')
            copytree('./models/'+model_name,'./checkpoint/run'+model_name+'/')
            time.sleep(10)
            gpt2.load_gpt2(sess,run_name="run"+model_name)
            x = False
        else:
            fil = input('What files would you like to tune based on? (full path) (txt files only): ')

            rounds_tmp = int(input('For how many rounds? (int): ').rstrip())
            if rounds_tmp:
                rounds = rounds_tmp
            print("Reading "+ fil.replace(".txt",'') +" " + str(rounds) + " times.")
            try:
                try:
                    gpt2.reset_session(sess)
                    sess = gpt2.start_tf_sess()
                    gpt2.load_gpt2(sess,run_name="run"+model_name)
                except Exception:
                    pass
                gpt2.finetune(sess, fil, model_name=model_name, steps=int(rounds),run_name="run"+model_name)
            except Exception:
                x = True
                
                print('It seems that the checkpoint files and the selected model do not match, should I purge the older model?')
                while x:
                    response = input('Purge? (y/n): ').lower()
                    if response == 'y':
                        shutil.rmtree('./checkpoint/run'+model_name+'/', ignore_errors=False, onerror=None)
                        # os.makedirs('./checkpoint/run'+model_name+'/')
                        copytree('./models/'+model_name,'./checkpoint/run'+model_name+'/')
                        time.sleep(10)
                        x = False
                        try:
                            gpt2.reset_session(sess)
                            sess = gpt2.start_tf_sess()
                            gpt2.finetune(sess, fil, model_name=model_name, steps=int(rounds),run_name="run"+model_name)
                        except Exception:
                            print('File from downlod repository seems to be corrupted. Should I get a new copy?')
                            y = True
                            while y:
                                response = input('Download (y/n)')
                                if response == 'y':
                                    gpt2.download_gpt2(model_name=model_name)
                                    gpt2.finetune(sess, fil, model_name=model_name, steps=int(rounds),run_name="run"+model_name)
                                    y = False
                                elif response == 'n':
                                    gpt2.reset_session(sess)
                                    gpt2.finetune(sess, fil, model_name=model_name, steps=int(rounds),run_name="run"+model_name)
                                    y = False
                                else:
                                    print('(y/n) please')
                        print("Generating text... this might take a while:")
                        gpt2.generate(sess, truncate='<|endoftext|>')
                    elif response == 'n':
                        print('Alright I can only generate then, using the older model saved in checkpoint.')
                        x = False
                        print("Generating text... this might take a while:")
                        gpt2.generate(sess, truncate='<|endoftext|>')
                    else:
                        print('(y/n) please')

    elif response == 'n':
        print('Got it, Generating only then')
        if not sess_flag:
            # gpt2.load_gpt2(sess)
            gpt2.load_gpt2(sess,run_name="run"+model_name)
        x = False
        prompt = input('Prompt: ')
        print('... ')
        gpt2.generate(sess, prefix=prompt, truncate='<|endoftext|>', include_prefix=False)
    else:
        print('(y/n) please')


while True:
    response = input('New prompt? (y/n): ').lower()
    if response == 'y':
        prompt = input('Prompt: ')
        print('... ')
        gpt2.generate(sess, prefix=prompt, truncate='<|endoftext|>', include_prefix=False)
    elif response == 'n':
        print('Got it Shutting down')
        exit(0)
    else:
        print('(y/n) please')

