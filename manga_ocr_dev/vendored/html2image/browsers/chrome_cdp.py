from .browser import CDPBrowser
from .search_utils import find_chrome

import os
import subprocess
from pathlib import Path

import requests
import json
from websocket import create_connection
import base64
import time

import websocket
# websocket.enableTrace(True)


class ChromeCDP(CDPBrowser):

    def __init__(
        self, executable=None, flags=None,
        print_command=False, cdp_port=9222,
        disable_logging=False,
    ):
        self.executable = executable
        if not flags:
            # for some reason, default-background-color prevents
            # the browser from running
            self.flags = [
                '--hide-scrollbars',
            ]
        else:
            self.flags = [flags] if isinstance(flags, str) else flags

        self.print_command = print_command
        self.cdp_port = cdp_port
        self._disable_logging = disable_logging

        self._ws = None  # Websocket connection
        self.proc = None  # Headless browser Popen object

        self.__id = 0

    @property
    def executable(self):
        return self._executable

    @executable.setter
    def executable(self, value):
        self._executable = find_chrome(value)

    @property
    def disable_logging(self):
        return self._disable_logging

    @disable_logging.setter
    def disable_logging(self, value):
        self._disable_logging = value

    @property
    def ws(self):
        if self._ws:
            return self._ws

        for i in range(20):
            try:
                r = requests.get(f'http://localhost:{self.cdp_port}/json', timeout=1)
                if r.status_code == 200:
                    pages = r.json()
                    page = next((p for p in pages if p.get('type') == 'page'), None)
                    if page and 'webSocketDebuggerUrl' in page:
                        self._ws = create_connection(page['webSocketDebuggerUrl'])
                        return self._ws
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, IndexError, StopIteration):
                time.sleep(0.5)

        raise RuntimeError(f"Could not connect to Chrome DevTools Protocol on port {self.cdp_port} after 10 seconds.")

    @property
    def _id(self):
        self.__id += 1
        return self.__id

    def cdp_send(self, method, **params):
        """
        """
        return self.ws.send(
            json.dumps({
                'id': self._id,
                'method': method,
                'params': params,
            })
        )

    def screenshot(
        self,
        input,
        output_path,
        output_file='screenshot.png',
        size=(1920, 1080),
    ):
        """
        """
        # Useful documentation about the Chrome DevTools Protocol:
        # https://chromedevtools.github.io/devtools-protocol/

        # "Enabling" the page allows to receive the Page.loadEventFired event
        self.cdp_send('Page.enable')

        self.cdp_send('Page.navigate', url=Path(input).as_uri())

        # Wait for page to load entirely
        while True:
            message = json.loads(self.ws.recv())
            method = message.get('method')
            if method == 'Page.loadEventFired':
                break

        self.cdp_send('Page.disable')

        self.cdp_send(
            'Emulation.setDeviceMetricsOverride',
            width=size[0],
            height=size[1],
            deviceScaleFactor=0,  # 0 disables the override
            mobile=False,
        )

        self.cdp_send(
            'Page.captureScreenshot',
            # captureBeyondViewport=True,
            # clip={
            #     'width': size[0],
            #     'height': size[1],
            #     'x': 500,
            #     'y': 200,
            #     'scale': 4
            # }
        )

        # get screenshot data when ready,
        # while potentially skipping unneeded messages
        while True:
            message = json.loads(self.ws.recv())
            # todo capture and display errors ?
            if 'result' in message and 'data' in message['result']:
                # retrive base64 encoded image data
                img_data = message['result']['data']
                break

        # Decode and write image data to file
        with open(os.path.join(output_path, output_file), 'wb') as f:
            f.write(base64.b64decode(img_data))

    def screenshot_as_bytes(
        self,
        input,
        size=(1920, 1080),
    ):
        """
        """
        # Useful documentation about the Chrome DevTools Protocol:
        # https://chromedevtools.github.io/devtools-protocol/

        # "Enabling" the page allows to receive the Page.loadEventFired event
        self.cdp_send('Page.enable')

        self.cdp_send('Page.navigate', url=Path(input).as_uri())

        # Wait for page to load entirely
        while True:
            message = json.loads(self.ws.recv())
            method = message.get('method')
            if method == 'Page.loadEventFired':
                break

        self.cdp_send('Page.disable')

        self.cdp_send(
            'Emulation.setDeviceMetricsOverride',
            width=size[0],
            height=size[1],
            deviceScaleFactor=0,  # 0 disables the override
            mobile=False,
        )

        self.cdp_send(
            'Runtime.evaluate',
            expression="document.documentElement.style.background = 'transparent'; document.body.style.background = 'transparent';"
        )

        self.cdp_send(
            'Page.captureScreenshot',
        )

        # get screenshot data when ready,
        # while potentially skipping unneeded messages
        while True:
            message = json.loads(self.ws.recv())
            # todo capture and display errors ?
            if 'result' in message and 'data' in message['result']:
                # retrive base64 encoded image data
                img_data = message['result']['data']
                break

        # Decode and return image data
        return base64.b64decode(img_data)

    def get_page_infos(self):
        """
        """
        self.cdp_send('Page.getLayoutMetrics')

        while True:
            message = json.loads(self.ws.recv())
            if 'result' in message and 'layoutViewport' in message['result']:
                return message['result']

    def print_pdf(self):
        # TODO : Page.printToPDF
        pass

    def __enter__(self):
        """
        """
        if not self.disable_logging:
            print(
                'Starting headless Chrome with '
                f'--remote-debugging-port={self.cdp_port}.'
            )

        self.flags.append('--remote-allow-origins=*')

        command = [
            f'{self.executable}',
            '--window-size=1920,1080',
            f'--remote-debugging-port={self.cdp_port}',
            '--headless=new',
            '--no-first-run',
            '--no-default-browser-check',
            *self.flags,
        ]

        if self.print_command:
            print(' '.join(command))

        self.proc = subprocess.Popen(command)

    def __exit__(self, *exc):
        """
        """
        if not self.disable_logging:
            print(f'Closing headless Chrome instance on port {self.cdp_port}.')

        # check if the process is still running
        if self.proc.poll() is None:
            # ensure that it is properly killed
            try:
                self.cdp_send('Browser.close')
                self.ws.close()
            except Exception:
                if not self.disable_logging:
                    print('Could not properly close the CDP and WebSocket connections.')

            try:
                self.proc.terminate()
            except Exception:
                if not self.disable_logging:
                    print('Could not properly kill Chrome.')