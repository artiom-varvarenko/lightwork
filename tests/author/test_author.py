import copy
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lightwork.author.sources import Source, load_source
from lightwork.author.workflow import AuthoringFailed, author, human_edit
from lightwork.config import Settings


@pytest.fixture
def server():
    state = {'responses': [], 'requests': [], 'pages': {}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/props':
                data, mime = json.dumps({'build_info': 'synthetic-test', 'model_path': '/tests/muse-glimmer.gguf', 'modalities': {'vision': True}}).encode(), 'application/json'
            elif self.path in state['pages']:
                data, mime = state['pages'][self.path]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            state['requests'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            reply = state['responses'].pop(0)
            data = json.dumps({'id': 'synthetic-test', 'choices': [{'message': {'content': reply}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 12, 'completion_tokens': 30}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state, f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    thread.join()
    httpd.server_close()


@pytest.fixture
def draft():
    return {'title': 'Synthetic test', 'objects': [{'id': 'cup', 'noun': 'red cup', 'kind': 'item', 'rows': None, 'cols': None, 'naming': None, 'channel': None}], 'steps': [{'id': 'place', 'say': 'Place cup.', 'detail': 'Keep it upright.', 'targets': ['cup'], 'arrow_from': None, 'arrow_to': None, 'checks': [{'id': 'visible', 'kind': 'judge', 'object': None, 'region': None, 'cells': [], 'n': None, 'direction': None, 'min_delta': None, 'ratio': None, 'tol': None, 'question': 'Is the cup upright?', 'required': True}], 'recovery': []}], 'inserted_steps': [], 'vocabulary': ['cup']}


def settings(tmp_path, base):
    return Settings(root=tmp_path, chat_base_url=base+'/v1', chat_model='muse-glimmer')


def source():
    return Source({'kind': 'text', 'ref': 'tests/original.txt', 'licence': 'original', 'attribution': 'Test author'}, 'Place cup.')


def test_source_html_and_attribution(server):
    state, base = server
    state['pages']['/guide'] = (b'<html><nav>Ignore menu</nav><script>private code</script><article><h1>Pack</h1><p>Place cup.</p></article></html>', 'text/html')
    result = load_source(base+'/guide', licence='CC BY 4.0', attribution='Author, https://example.org/source; adapted', privacy_confirmed=True)
    assert result.text == 'Pack\nPlace cup.'
    assert result.metadata['licence'] == 'CC BY 4.0'
    assert 'Author' in result.metadata['attribution']


@pytest.mark.parametrize('kwargs', [dict(privacy_confirmed=False), dict(licence='CC BY-NC-SA 3.0'), dict(attribution='')])
def test_source_guards_before_network(kwargs):
    args = dict(licence='original', attribution='Author', privacy_confirmed=True)
    args.update(kwargs)
    with pytest.raises(ValueError):
        load_source('http://127.0.0.1:1/never-contacted', **args)


def test_ifixit_rejected():
    with pytest.raises(ValueError, match='iFixit'):
        load_source('https://www.ifixit.com/Guide/example/1', licence='original', attribution='Author', privacy_confirmed=True)


def simple_pdf():
    stream = b'BT /F1 12 Tf 30 70 Td (Place red cup.) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>', b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>', b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>', b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
    data = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f'{index} 0 obj\n'.encode()+obj+b'\nendobj\n')
    start = len(data)
    data.extend(b'xref\n0 6\n0000000000 65535 f \n')
    for offset in offsets[1:]:
        data.extend(f'{offset:010} 00000 n \n'.encode())
    data.extend(f'trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n{start}\n%%EOF\n'.encode())
    return bytes(data)


def test_pdf_local_and_http(server, tmp_path):
    if not shutil.which('pdftotext'):
        pytest.skip('Poppler not installed')
    state, base = server
    raw = simple_pdf()
    state['pages']['/guide.pdf'] = (raw, 'application/pdf')
    path = tmp_path/'guide.pdf'
    path.write_bytes(raw)
    for ref in (str(path), base+'/guide.pdf'):
        result = load_source(ref, licence='original', attribution='Test author', privacy_confirmed=True)
        assert result.text == 'Place red cup.'
        assert result.pdf == raw
        assert result.metadata['kind'] == 'pdf'


def test_semantic_repair_records_and_human_diff(server, tmp_path, draft):
    state, base = server
    invalid = copy.deepcopy(draft)
    invalid['steps'][0]['targets'] = ['missing-object']
    state['responses'] = [json.dumps(invalid), json.dumps(draft)]
    output = tmp_path/'synthetic-pack'
    graph = author(source(), slug='synthetic-pack', output=output, settings=settings(tmp_path, base), backend='glimmer')
    assert graph.source.attribution == 'Test author'
    assert len(state['requests']) == 2
    repair = json.dumps(state['requests'][1]['messages'])
    assert 'missing-object' in repair
    assert len(list(output.glob('authoring/*/calls/*/response.json'))) == 2
    edited = graph.model_dump(mode='json', by_alias=True)
    edited['steps'][0]['detail'] = 'Human clarification.'
    edit_file = tmp_path/'edit.json'
    edit_file.write_text(json.dumps(edited))
    human_edit(output, edit_file)
    assert 'Human clarification.' in next((output/'authoring').glob('human-*.diff')).read_text()
    assert json.loads(next(output.glob('authoring/*/model-stepgraph.json')).read_text())['steps'][0]['detail'] == 'Keep it upright.'


def test_two_bad_outputs_preserved_without_graph(server, tmp_path):
    state, base = server
    state['responses'] = ['not json', 'still not json']
    output = tmp_path/'failed'
    with pytest.raises(AuthoringFailed):
        author(source(), slug='failed', output=output, settings=settings(tmp_path, base), backend='glimmer')
    assert len(state['requests']) == 2
    assert not (output/'stepgraph.json').exists()
    assert len(list(output.glob('authoring/*/calls/*/response.json'))) == 2
    assert json.loads(next(output.glob('authoring/*/result.json')).read_text())['status'] == 'error'


def test_existing_graph_not_overwritten(tmp_path):
    (tmp_path/'stepgraph.json').write_text('preserve me')
    with pytest.raises(ValueError, match='exists'):
        author(source(), slug='pack', output=tmp_path, settings=Settings(root=tmp_path), backend='meta')
    assert (tmp_path/'stepgraph.json').read_text() == 'preserve me'
