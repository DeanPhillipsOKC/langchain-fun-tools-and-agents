from langchain.callbacks.base import BaseCallbackHandler

class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwards):
        print(messages)