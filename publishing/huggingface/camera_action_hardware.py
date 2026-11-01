"""Small public hardware context; hostnames, serials and arbitrary fields excluded."""
import re

MAX_BYTES = 16 * 1024
FIELDS = {'scope', 'system', 'release', 'machine', 'python', 'versions', 'hardware', 'native_workers', 'gpu_use'}
HARDWARE = {'hw.model', 'hw.memsize', 'hw.physicalcpu', 'hw.logicalcpu', 'machdep.cpu.brand_string'}


def validate_hardware(value):
    if type(value) is not dict or set(value) != FIELDS:
        raise ValueError('Hardware context must use the public field whitelist; no hostname, serial or account fields.')
    for key in ('scope', 'system', 'release', 'machine', 'python', 'gpu_use'):
        text=value[key]
        if type(text) is not str or not 1 <= len(text) <= (1024 if key in ('scope','gpu_use') else 80) or any(ord(c)<32 for c in text):
            raise ValueError('Hardware context contains invalid or unbounded text.')
    if type(value['native_workers']) is not int or not 1 <= value['native_workers'] <= 1024:
        raise ValueError('Hardware context requires a bounded native worker count.')
    versions=value['versions']; hardware=value['hardware']
    if type(versions) is not dict or set(versions) != {'numpy','mujoco'} or type(hardware) is not dict or set(hardware) != HARDWARE:
        raise ValueError('Hardware/software details must use the public field whitelist.')
    for version in [value['python'],*versions.values()]:
        if type(version) is not str or not re.fullmatch(r'[0-9][0-9A-Za-z.+_-]{0,79}',version):
            raise ValueError('Software version must be a plain bounded version string.')
    for key in ('hw.memsize','hw.physicalcpu','hw.logicalcpu'):
        number=hardware[key]
        if type(number) is not str or not re.fullmatch(r'[1-9][0-9]{0,18}',number) or int(number)>(2**60 if key=='hw.memsize' else 4096):
            raise ValueError('Hardware memory and CPU counts must be positive bounded decimal strings.')
    for key in ('hw.model','machdep.cpu.brand_string'):
        text=hardware[key]
        if type(text) is not str or not 1<=len(text)<=120 or any(ord(c)<32 for c in text) or any(c in text for c in ('/','\\','@')):
            raise ValueError('Hardware product name must be a bounded public model identifier.')
    return value
