########################################################################
# DistributorApp work as an HTTPServer accept the requests from TrainingWorkers
# - GET to respond a training task via MIME resp, including
#        a) the JSON model definition of the agent._brain
#        b) the recent and version-ed weights of the agent._brain
#        c) a sample batch from agent._sampleBatches()
# - POST to accept a training result via MIME body, including
#        a) the weights result of training
#        b) the version when the worker took the task
#        As the response to POST, a new task like above GET's response would be delivered to the worker
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from Application import BaseApplication
import cgi
import re

class DistributorApp(BaseHTTPRequestHandler, BaseApplication):

    URI_STACK=[
        (re.compile('^/sim/task'), __get_SimulatorTask, __post_SimulatorResult),
        (re.compile('^/train/task'), __get_TrainingTask, __post_TrainingResult)
    ]
    
    def __init__(self, config, duration, errors, *args, **kwargs):
        self._config = config
        self._duration = duration
        self._errors = errors
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs) 
        BaseApplication.__init__(self, *args, **kwargs) 

        # self.extensions_map.update({
        #     '.webapp': 'application/x-web-app-manifest+json',
        # });

    def do_GET(self):
        for uristk in DistributorApp.URI_STACK:
            m = uristk[0].match(self.path)
            if m :
                return self.uristk[1]()

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        func = None
        for uristk in DistributorApp.URI_STACK:
            m = uristk[0].match(self.path)
            if m :
                return self.uristk[2]()

        self.send_response(404)
        self.end_headers()

    def __post_SimulatorResult(self):
        self.send_response(200)
        self.end_headers()

    def __get_SimulatorTask(self):
        self.send_response(200)
        self.end_headers()

    def __get_TrainingTask(self):
        self.send_response(200)
        self.end_headers()

    def __post_TrainingResult(self):
        self.send_response(200)
        self.end_headers()

    def template_do_POST(self):
        # Parse the form data posted
        form = cgi.FieldStorage(
            fp=self.rfile, 
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })

        # Begin the response
        self.send_response(200)
        self.end_headers()
        self.wfile.write('Client: %s\n' % str(self.client_address))
        self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
        self.wfile.write('Path: %s\n' % self.path)
        self.wfile.write('Form data:\n')

        # Echo back information about what was posted in the form
        for field in form.keys():
            field_item = form[field]
            if field_item.filename:
                # The field contains an uploaded file
                file_data = field_item.file.read()
                file_len = len(file_data)
                del file_data
                self.wfile.write('\tUploaded %s as "%s" (%d bytes)\n' % \
                        (field, field_item.filename, file_len))
            else:
                # Regular form value
                self.wfile.write('\t%s=%s\n' % (field, form[field].value))
        return

    def template_do_GET(self):
        parsed_path = urlparse.urlparse(self.path)
        message_parts = [
                'CLIENT VALUES:',
                'client_address=%s (%s)' % (self.client_address,
                                            self.address_string()),
                'command=%s' % self.command,
                'path=%s' % self.path,
                'real path=%s' % parsed_path.path,
                'query=%s' % parsed_path.query,
                'request_version=%s' % self.request_version,
                '',
                'SERVER VALUES:',
                'server_version=%s' % self.server_version,
                'sys_version=%s' % self.sys_version,
                'protocol_version=%s' % self.protocol_version,
                '',
                'HEADERS RECEIVED:',
                ]
        for name, value in sorted(self.headers.items()):
            message_parts.append('%s=%s' % (name, value.rstrip()))
        message_parts.append('')
        message = '\r\n'.join(message_parts)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(message)
        return


class HpHttpServer(HTTPServer):

    stopped = False
    allow_reuse_address = True

    def __init__(self, *args, **kw):
        super(HpHttpServer, self).__init__(self, *args, **kw)
        self.register_function(lambda: 'OK', 'ping')

    def serve_forever(self):
        while not self.stopped:
            self.handle_request()

    def force_stop(self):
        self.server_close()
        self.stopped = True
        self.create_dummy_request()

    def create_dummy_request(self):
        conn = xmlrpclib.Server('http://%s:%s' % self.server_address)
        conn.ping()

'''
Notes: Averaging the weights

https://stackoverflow.com/questions/48212110/average-weights-in-keras-models

weights = [model.get_weights() for model in models]
Now - create a new averaged weights:

new_weights = list()

for weights_list_tuple in zip(*weights):
    new_weights.append(
        [numpy.array(weights_).mean(axis=0)\
            for weights_ in zip(*weights_list_tuple)])


And what is left is to set these weights in a new model:

new_model = new_model.set_weights(new_weights)
'''