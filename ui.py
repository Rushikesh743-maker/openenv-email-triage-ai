import gradio as gr
from env.environment import EmailEnv
from tasks.easy import EasyTask

env = EmailEnv(EasyTask())

def run():
    return str(env.reset())

gr.Interface(fn=run, inputs=[], outputs="text").launch()
