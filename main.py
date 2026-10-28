from pyfedappwrap.engine.runtime import FedDBEngine

from app import OneHotEncodingAPP

engine = FedDBEngine()
engine.register(OneHotEncodingAPP())

if __name__ == '__main__':
    engine.start()
    engine.wait_until_stop()