"""No-network transport tests; original publisher identity checks stay separate."""
import io
from email.message import Message
from unittest import TestCase, main, mock
from urllib.error import HTTPError, URLError
from urllib.request import Request
import ssl

import grip_clock_public_http as direct
from publish import PublicationError


OID = 'a' * 40
REPO = direct.pub.REPOS['dataset']
URL = f'https://huggingface.co/datasets/{REPO}/resolve/{OID}/case.json?download=true'


class Response(io.BytesIO):
    def __init__(self, raw=b'abc', *, status=200, url=URL, headers=None):
        super().__init__(raw)
        self.status, self.url = status, url
        self.headers = Message()
        for key, value in headers or []:
            self.headers[key] = value

    def geturl(self):
        return self.url


class PublicTransportTests(TestCase):
    def get(self, response=None, *, effects=None):
        opener = mock.Mock()
        opener.open.side_effect = effects if effects is not None else [response or Response()]
        with mock.patch.object(direct, '_opener', return_value=opener), mock.patch.object(direct.time, 'sleep') as sleep:
            raw = direct.download(REPO, 'case.json', 'dataset', OID)
        return raw, opener, sleep

    def test_fixed_urls_and_anonymous_identity_headers(self):
        raw, opener, _ = self.get(Response(headers=[('Content-Length', '3')]))
        self.assertEqual(raw, b'abc')
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, URL)
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual({key.lower(): value for key, value in request.header_items()},
                         {key.lower(): value for key, value in direct.HEADERS.items()})
        self.assertEqual(opener.open.call_args.kwargs, {'timeout': 30})
        for kind, prefix in direct.ROOTS.items():
            self.assertIn('/'+prefix+direct.pub.REPOS[kind]+'/resolve/',
                          direct._public_url(direct.pub.REPOS[kind], 'README.md', kind, OID))

    def test_rejects_foreign_repositories_unpinned_revisions_and_paths_before_network(self):
        invalid = [('other/repo', 'case.json', 'dataset', OID),
                   (REPO, 'case.json', 'wrong', OID),
                   (REPO, 'case.json', 'dataset', 'main'),
                   (REPO, 'case.json', 'dataset', 'A'*40),
                   (REPO, '../case.json', 'dataset', OID),
                   (REPO, '/case.json', 'dataset', OID),
                   (REPO, 'case.json?x=1', 'dataset', OID),
                   (REPO, 'https://host/a.json', 'dataset', OID)]
        with mock.patch.object(direct, '_opener') as opener:
            for values in invalid:
                with self.subTest(values=values), self.assertRaises(PublicationError):
                    direct.download(*values)
            opener.assert_not_called()
        self.assertIn('/a%252Fb.json?', direct._public_url(REPO, 'a%2Fb.json', 'dataset', OID))

    def test_redirect_allowlist_and_fixed_headers(self):
        handler = direct._PublicRedirect()
        for url in ['https://huggingface.co/a', 'https://cdn-lfs.hf.co/a?signature=x',
                    'https://cas-bridge.xethub.hf.co/a']:
            request = handler.redirect_request(Request(URL, headers={'Cookie':'bad','Authorization':'bad'}),
                                               None, 302, '', {}, url)
            self.assertEqual(request.get_method(), 'GET')
            self.assertEqual({key.lower() for key, _ in request.header_items()},
                             {'user-agent','accept-encoding'})
        for url in ['http://huggingface.co/a', 'https://evil.test/a',
                    'https://huggingface.co.evil.test/a', 'https://evilhf.co/a',
                    'https://user@huggingface.co/a', 'https://huggingface.co:443/a',
                    'https://huggingface.co/a#fragment', 'https://huggingface.co./a',
                    'https://huggingface.co/a\nInjected', 'https://[broken/a']:
            with self.subTest(url=url), self.assertRaises(PublicationError):
                handler.redirect_request(Request(URL), None, 302, '', {}, url)

    def test_opener_has_no_proxy_cookie_or_auth_handlers_and_verified_tls(self):
        with mock.patch.dict('os.environ', {'HTTPS_PROXY':'http://unusable.invalid:1'}):
            opener = direct._opener()
        classes = {type(handler).__name__ for handler in opener.handlers}
        self.assertFalse(classes & {'ProxyHandler','HTTPCookieProcessor','HTTPBasicAuthHandler',
                                    'HTTPDigestAuthHandler','ProxyBasicAuthHandler','ProxyDigestAuthHandler'})
        self.assertEqual(opener.addheaders, [])
        tls = next(handler for handler in opener.handlers if type(handler).__name__ == 'HTTPSHandler')
        self.assertTrue(tls._context.check_hostname)
        self.assertEqual(tls._context.verify_mode, ssl.CERT_REQUIRED)

    def test_redirect_closes_unread_body_and_preserves_timeout(self):
        handler = direct._PublicRedirect()
        handler.parent = mock.Mock()
        request = Request(URL)
        request.timeout = 30
        body = mock.Mock()
        handler.http_error_302(request,body,302,'',{'Location':'https://cdn-lfs.hf.co/blob'})
        body.close.assert_called_once()
        body.read.assert_not_called()
        following = handler.parent.open.call_args.args[0]
        self.assertEqual(following._dvidia_redirects,1)
        self.assertEqual(handler.parent.open.call_args.kwargs,{'timeout':30})

    def test_redirect_chain_bound_and_unsafe_location_close_without_following(self):
        handler = direct._PublicRedirect()
        handler.parent = mock.Mock()
        request = Request(URL)
        request.timeout = 30
        request._dvidia_redirects = 5
        for location in ['https://cdn-lfs.hf.co/blob','https://evil.invalid/blob',
                         'https://huggingface.co/a\nb']:
            body = mock.Mock()
            with self.subTest(location=location), self.assertRaises(PublicationError):
                handler.http_error_302(request,body,302,'',{'Location':location})
            body.close.assert_called_once()
            body.read.assert_not_called()
        handler.parent.open.assert_not_called()

    def test_success_without_declared_length_and_maximum_bounded_read(self):
        response = Response(b'x' * 8)
        with mock.patch.object(direct, 'MAX_BYTES', 8):
            self.assertEqual(self.get(response)[0], b'x'*8)
            with self.assertRaises(PublicationError):
                self.get(Response(b'x'*9))

    def test_rejects_declared_oversize_invalid_duplicate_and_truncated_lengths(self):
        for headers, raw in [([('Content-Length','8388609')],b''),
                             ([('Content-Length','-1')],b'abc'),
                             ([('Content-Length','many')],b'abc'),
                             ([('Content-Length','9'*5000)],b'abc'),
                             ([('Content-Length','3'),('Content-Length','3')],b'abc'),
                             ([('Content-Length','4')],b'abc'),
                             ([('Content-Length','2')],b'abc')]:
            with self.subTest(headers=headers), self.assertRaises(PublicationError):
                self.get(Response(raw,headers=headers))

    def test_rejects_other_encoding_non200_and_foreign_final_response(self):
        for response in [Response(headers=[('Content-Encoding','gzip')]),
                         Response(headers=[('Content-Encoding','br')]), Response(status=206),
                         Response(status=204), Response(url='https://elsewhere.invalid/case.json')]:
            with self.subTest(response=response), self.assertRaises(PublicationError):
                self.get(response)

    def test_transient_get_retries_are_bounded_and_backoff_is_bounded(self):
        effects=[HTTPError(URL,503,'ignored',{},io.BytesIO()) for _ in range(3)]+[Response()]
        raw, opener, sleep = self.get(effects=effects)
        self.assertEqual(raw,b'abc')
        self.assertEqual(opener.open.call_count,4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],[.25,.5,1.])
        self.assertTrue(all(call.args[0].get_method()=='GET' for call in opener.open.call_args_list))
        with mock.patch.object(direct,'_opener') as build, mock.patch.object(direct.time,'sleep'):
            build.return_value.open.side_effect=URLError('unavailable')
            with self.assertRaises(PublicationError):
                direct.download(REPO,'case.json','dataset',OID)
            self.assertEqual(build.return_value.open.call_count,4)

    def test_nonretryable_status_and_tls_verification_fail_immediately(self):
        for error in [HTTPError(URL,404,'ignored',{},io.BytesIO()),
                      URLError(ssl.SSLCertVerificationError('invalid certificate'))]:
            with mock.patch.object(direct,'_opener') as build, mock.patch.object(direct.time,'sleep') as sleep:
                build.return_value.open.side_effect=error
                with self.assertRaises(PublicationError):
                    direct.download(REPO,'case.json','dataset',OID)
                self.assertEqual(build.return_value.open.call_count,1)
                sleep.assert_not_called()

    def test_invalid_response_is_not_retried_and_no_global_opener_is_used(self):
        with mock.patch.object(direct,'_opener') as build, mock.patch.object(direct.time,'sleep') as sleep:
            build.return_value.open.return_value=Response(headers=[('Content-Length','100')])
            with self.assertRaises(PublicationError):
                direct.download(REPO,'case.json','dataset',OID)
            self.assertEqual(build.return_value.open.call_count,1)
            sleep.assert_not_called()

    def test_safe_diagnostics_preserve_status_class_and_path_without_sensitive_details(self):
        error=HTTPError('https://cdn-lfs.hf.co/object?signed=do-not-retain',403,'secret body',{},io.BytesIO())
        with mock.patch.object(direct,'_opener') as build:
            build.return_value.open.side_effect=error
            with self.assertRaises(direct.PublicReadbackError) as caught:
                direct.download(REPO,'case.json','dataset',OID)
        self.assertEqual((caught.exception.path,caught.exception.http_status,caught.exception.failure_class,
                          caught.exception.attempts),('case.json',403,'HTTPError',1))
        self.assertNotIn('signed',str(caught.exception))
        self.assertNotIn('secret',str(caught.exception))
        with self.assertRaises(direct.PublicReadbackError) as mismatch:
            self.get(Response(b'abc',headers=[('Content-Length','4')]))
        self.assertEqual(mismatch.exception.http_status,200)
        self.assertEqual(mismatch.exception.check_code,'declared_length_mismatch')


if __name__ == '__main__':
    main()
