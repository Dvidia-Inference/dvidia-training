from copy import deepcopy
import unittest

from camera_action_hardware import validate_hardware


def fixture():
    return dict(scope='Shared host; offline simulator timings only.', system='Darwin', release='25.5.0',
        machine='arm64', python='3.12.9', versions={'numpy':'2.5.3','mujoco':'3.15.0'},
        hardware={'hw.model':'Mac17,4','hw.memsize':'25769803776','hw.physicalcpu':'10',
            'hw.logicalcpu':'10','machdep.cpu.brand_string':'Apple M5'},native_workers=1,
        gpu_use='Existing macOS CGL rendering; CPU policy and physics.')


class HardwareContextTests(unittest.TestCase):
    def test_public_hardware_is_accepted_and_private_fields_rejected(self):
        validate_hardware(fixture())
        for key in ('hostname','serial','account','machine_uuid'):
            for nested in (False,True):
                value=fixture(); target=value['hardware'] if nested else value;target[key]='private fixture'
                with self.subTest(key=key,nested=nested),self.assertRaisesRegex(ValueError,'whitelist'):
                    validate_hardware(value)

    def test_text_worker_counts_and_numeric_hardware_are_bounded(self):
        for key,invalid in (('scope','x'*1025),('python','/private/version'),('native_workers',True),('native_workers',0)):
            value=fixture();value[key]=invalid
            with self.subTest(key=key),self.assertRaises(ValueError):validate_hardware(value)
        for key,invalid in (('hw.memsize','0'),('hw.physicalcpu','4097'),('hw.logicalcpu',10),('hw.model','/Users/private')):
            value=fixture();value['hardware'][key]=invalid
            with self.subTest(key=key),self.assertRaises(ValueError):validate_hardware(value)


if __name__=='__main__':unittest.main()
