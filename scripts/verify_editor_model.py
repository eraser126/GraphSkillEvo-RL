"""Offline model-integrity check. Never instantiates model weights or calls APIs."""
import argparse,hashlib,json,struct,math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.model);info=json.loads(Path(a.metadata).read_text());files={}
    if info['id']!='Qwen/Qwen3-4B-Instruct-2507' or info['sha']!='cdbee75f17c01a7cc42f958dc650907174af0554':
        raise ValueError('Unexpected source model/revision')
    for row in info['siblings']:
        f=root/row['rfilename']
        if not f.is_file() or f.stat().st_size!=row['size']:raise ValueError('Missing/incomplete '+row['rfilename'])
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
        if row.get('lfs') and h.hexdigest()!=row['lfs']['sha256']:raise ValueError('Model shard hash mismatch')
        if not row.get('lfs'):
            git=hashlib.sha1(f"blob {row['size']}\0".encode()+f.read_bytes()).hexdigest()
            if git!=row['blobId']:raise ValueError('Official source blob mismatch: '+row['rfilename'])
        files[row['rfilename']]=dict(size=f.stat().st_size,sha256=h.hexdigest())
    config=json.loads((root/'config.json').read_text())
    assert config['model_type']=='qwen3'
    index=json.loads((root/'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):assert name in files
    observed={};data_bytes=0
    for name in set(index['weight_map'].values()):
        with (root/name).open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0]
            if n>16*1024*1024:raise ValueError('Unexpected safetensors header size')
            header=json.loads(stream.read(n))
        end=0
        for key,value in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),
                                key=lambda item:item[1]['data_offsets'][0]):
            start,stop=value['data_offsets']
            assert start==end and stop>=start,'Noncontiguous tensor offsets'
            assert value['dtype']=='BF16' and stop-start==2*math.prod(value['shape']),'Tensor shape/byte mismatch'
            end=stop
        assert end==(root/name).stat().st_size-8-n,'Truncated tensor data'
        observed.update({key:name for key in header if key!='__metadata__'})
        data_bytes+=(root/name).stat().st_size-8-n
    assert observed==index['weight_map'],'Index/shard tensor mismatch'
    # This pinned upstream index reports 655360 more bytes than the verified BF16
    # tensors contain. Aggregate metadata is not a completeness oracle: validate
    # official file hashes, every tensor mapping, shape and data offset above.
    aggregate_matches=data_bytes==index['metadata']['total_size']
    result=dict(model=info['id'],revision=info['sha'],files=files,weights_loaded=False,
                tensor_count=len(observed),actual_tensor_bytes=data_bytes,
                upstream_index_total_size=index['metadata']['total_size'],aggregate_metadata_matches=aggregate_matches,
                warnings=[] if aggregate_matches else ['Official index aggregate size differs from verified tensor data; upstream files kept unchanged.'],
                training_started=False,api_called=False)
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(model=result['model'],revision=result['revision'],files=len(files),
                         total_bytes=sum(x['size'] for x in files.values()),weights_loaded=False,training_started=False)))


if __name__=='__main__':main()
