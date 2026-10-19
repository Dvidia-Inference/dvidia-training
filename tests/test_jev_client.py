"""Provider contract tests use fake transports only; never require an API key."""
from copy import deepcopy
from email.parser import BytesParser
from email.policy import default
from hashlib import sha256
import io
import json
import socket
import struct
import subprocess
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
import zlib

from dvidia_training import jev_client as jev


def png(width=320, height=280):
    def chunk(kind, body):
        return (struct.pack('>I', len(body)) + kind + body
                + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * width * 3) * height))
            + chunk(b'IEND', b''))


def valid_response(present=.99, complete=.03, verdict='incomplete'):
    choices = {name: .01 for name in ('complete', 'incomplete', 'unknown')}
    choices[verdict] = .98
    return {'model': jev.MODEL,
            'answers': {'presence': {'type': 'noul', 'noul': present},
                        'completion': {'type': 'noul', 'noul': complete},
                        'verdict': {'type': 'choice', 'choice': verdict,
                                    'probabilities': choices, 'confidence': .97}},
            'usage': {'input_tokens': 1200, 'output_tokens': 20}}


class FakeResponse:
    status = 200
    def __init__(self, value, delay=0):
        self.body = json.dumps(value).encode()
        self.delay = delay
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def read(self, limit):
        time.sleep(self.delay)
        return self.body[:limit]


class JevContractTests(unittest.TestCase):
    def test_multipart_images_are_uploaded_once_and_referenced_in_state(self):
        current, goal = png(), png(321)
        state = {'support_force_n': .25}
        prepared = jev.build_goal_request(current, goal, task='Place the target shoe.', measured_state=state)
        message = BytesParser(policy=default).parsebytes(
            b'Content-Type: ' + prepared.content_type.encode() + b'\r\nMIME-Version: 1.0\r\n\r\n' + prepared.body)
        parts = list(message.iter_parts())
        self.assertEqual([p.get_param('name', header='Content-Disposition') for p in parts],
                         ['request', 'current', 'goal'])
        payload = json.loads(parts[0].get_payload(decode=True))
        self.assertEqual(payload['model'], 'jev-1.14.0')
        self.assertEqual(set(payload['questions']), {'presence', 'completion', 'verdict'})
        self.assertEqual(parts[1].get_payload(decode=True), current)
        self.assertEqual(parts[2].get_payload(decode=True), goal)
        self.assertEqual(set(payload['assets']), {payload['state']['current_image'], payload['state']['goal_image']})
        self.assertEqual(prepared.images['current']['sha256'], sha256(current).hexdigest())
        state['support_force_n'] = 999
        self.assertEqual(prepared.request['state']['measured_state']['support_force_n'], .25)
        self.assertEqual(prepared.request_sha256, sha256(prepared.body).hexdigest())
        repeated = jev.build_goal_request(current, goal, task='Place the target shoe.', measured_state={'support_force_n': .25})
        self.assertEqual(prepared.body, repeated.body)

    def test_rgb_omits_telemetry_and_rejects_invalid_images_and_nonfinite_state(self):
        prepared = jev.build_goal_request(png(), png(), task='Place the target shoe.')
        self.assertNotIn('measured_state', prepared.request['state'])
        for image in (b'not a PNG', png(279), png(1569), png()[:-12], png()[:-13] + b'x' * 13):
            with self.subTest(image_bytes=len(image)), self.assertRaises(ValueError):
                jev.build_goal_request(image, png(), task='Place shoe.')
        with self.assertRaises(ValueError):
            jev.build_goal_request(png(), png(), task='Place shoe.', measured_state={'force': float('nan')})

    def test_noul_uncertainty_is_preserved_independently_of_tristate_choice(self):
        result = jev.parse_goal_response(valid_response(.5, .8, 'unknown'))
        self.assertEqual(result['presence'], 'unknown')
        self.assertEqual(result['verdict'], 'unknown')
        self.assertEqual(result['p_complete'], .8)
        self.assertAlmostEqual(result['estimated_input_cost_usd'], .0000504)
        self.assertEqual(jev.parse_goal_response(valid_response(.2))['presence'], 'absent')
        self.assertEqual(jev.parse_goal_response(valid_response(.8))['presence'], 'present')

    def test_invalid_model_probabilities_choice_and_usage_are_not_scored(self):
        variants = []
        row = valid_response(); row['model'] = 'jev-latest'; variants.append(row)
        row = valid_response(); row['answers']['presence']['noul'] = True; variants.append(row)
        row = valid_response(); row['answers']['completion']['noul'] = float('nan'); variants.append(row)
        row = valid_response(); row['answers']['verdict']['probabilities'].pop('unknown'); variants.append(row)
        row = valid_response(); row['answers']['verdict']['probabilities']['complete'] = .9; variants.append(row)
        row = valid_response(); row['answers']['verdict']['choice'] = 'complete'; variants.append(row)
        row = valid_response(); row['usage']['input_tokens'] = '1200'; variants.append(row)
        for response in variants:
            with self.subTest(response=response), self.assertRaises(ValueError):
                jev.parse_goal_response(response)

    def test_success_receipt_times_upload_response_and_keeps_credentials_private(self):
        client = jev.JevClient('secret-for-test', timeout_s=3)
        client._opener = Mock()
        client._opener.open.return_value = FakeResponse(valid_response(), delay=.003)
        receipt = client.evaluate(png(), png(), task='Place shoe.')
        self.assertEqual(receipt['status'], 'ok')
        self.assertEqual(receipt['attempts'], 1)
        self.assertGreaterEqual(receipt['latency_ms']['http'], 2)
        self.assertGreaterEqual(receipt['latency_ms']['total'], receipt['latency_ms']['http'])
        request = client._opener.open.call_args.args[0]
        self.assertEqual(request.full_url, jev.ENDPOINT)
        self.assertEqual(request.get_header('Typesafe-danger-zone-images'), 'true')
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret-for-test')
        self.assertEqual(client._opener.open.call_args.kwargs['timeout'], 3)
        self.assertNotIn('secret-for-test', json.dumps(receipt))
        self.assertNotIn('Authorization', json.dumps(receipt))

    def test_missing_auth_never_calls_provider_and_is_not_an_absent_prediction(self):
        client = jev.JevClient('')
        client._opener = Mock()
        result = client.evaluate(png(), png(), task='Place shoe.')
        client._opener.open.assert_not_called()
        self.assertEqual(result['status'], 'missing_credentials')
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['presence'], 'unknown')
        self.assertIsNone(result['p_complete'])
        self.assertIsNone(result['estimated_input_cost_usd'])
        self.assertIsNone(result['latency_ms']['http'])

    def test_timeout_and_rate_limit_are_single_attempts_and_keep_error_evidence(self):
        for error in (socket.timeout('timed out'), URLError(socket.timeout('timeout')),
                      HTTPError(jev.ENDPOINT, 429, 'limit', {}, io.BytesIO(b'{"error":"secret-for-test"}'))):
            client = jev.JevClient('secret-for-test')
            client._opener = Mock()
            client._opener.open.side_effect = error
            result = client.evaluate(png(), png(), task='Place shoe.')
            self.assertEqual(client._opener.open.call_count, 1)
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['verdict'], 'unknown')
            self.assertIsNone(result['p_complete'])
            self.assertNotIn('secret-for-test', json.dumps(result))
            if isinstance(error, HTTPError):
                self.assertEqual(result['http_status'], 429)
                self.assertEqual(result['response']['error'], '[REDACTED]')
            else:
                self.assertEqual(result['error']['kind'], 'timeout')

    def test_invalid_provider_response_preserved_but_not_scored(self):
        client = jev.JevClient('secret-for-test')
        client._opener = Mock()
        response = valid_response()
        response['answers']['presence']['noul'] = 'secret-for-test'
        client._opener.open.return_value = FakeResponse(response)
        result = client.evaluate(png(), png(), task='Place shoe.')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error']['kind'], 'invalid_response')
        self.assertIsNone(result['p_present'])
        self.assertNotIn('secret-for-test', json.dumps(result))
        self.assertEqual(result['response']['answers']['presence']['noul'], '[REDACTED]')
        self.assertEqual(result['usage']['input_tokens'], 1200)
        self.assertAlmostEqual(result['estimated_input_cost_usd'], .0000504)

    def test_invalid_usage_is_unknown_cost_and_http_error_usage_is_not_lost(self):
        client = jev.JevClient('secret-for-test')
        client._opener = Mock()
        response = valid_response()
        response['usage']['input_tokens'] = -1
        client._opener.open.return_value = FakeResponse(response)
        result = client.evaluate(png(), png(), task='Place shoe.')
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['usage'])
        self.assertIsNone(result['estimated_input_cost_usd'])
        body = json.dumps({'error': 'model failed', 'usage': {'input_tokens': 400, 'output_tokens': 0}}).encode()
        client._opener.open.side_effect = HTTPError(jev.ENDPOINT, 500, 'failed', {}, io.BytesIO(body))
        result = client.evaluate(png(), png(), task='Place shoe.')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['usage']['input_tokens'], 400)
        self.assertAlmostEqual(result['estimated_input_cost_usd'], .0000168)

    def test_redirect_is_refused_before_credential_can_cross_hosts(self):
        with self.assertRaises(HTTPError):
            jev._NoRedirect().redirect_request(Mock(full_url=jev.ENDPOINT), None, 307,
                                               'Redirect', {}, 'https://elsewhere.invalid/')

    def test_keychain_only_queried_when_explicitly_named_and_env_takes_precedence(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(jev.subprocess, 'run') as run:
            self.assertIsNone(jev.load_api_key())
            run.assert_not_called()
            with self.assertRaises(ValueError):
                jev.load_api_key(keychain_service='dvidia-typesafe')
            run.return_value = subprocess.CompletedProcess([], 0, 'secret-for-test\n', '')
            self.assertEqual(jev.load_api_key(keychain_service='dvidia-typesafe', keychain_account='dvidia-research'), 'secret-for-test')
            self.assertEqual(run.call_args.args[0], ['/usr/bin/security', 'find-generic-password', '-s', 'dvidia-typesafe', '-a', 'dvidia-research', '-w'])
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'env-secret'}), patch.object(jev.subprocess, 'run') as run:
            self.assertEqual(jev.load_api_key(keychain_service='dvidia-typesafe', keychain_account='dvidia-research'), 'env-secret')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
