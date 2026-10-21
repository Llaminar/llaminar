"""Optional fixed-history corpus adapter for the existing native canary lifecycle.

This module never starts a model process or owns driver/resource retirement.
The existing runner owns those operations. Source-owned argv/config, explicit
paired arithmetic/document obligations and actual native proofs stay separate
from numerical acceptance and unexecuted benchmark obligations.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json

import model_kld_corpus_v2 as corpus_v2
from model_kld_corpus import identity, metadata_digest, require, unchanged

SELECTORS = {
    'gdn_projection': 'LLAMINAR_GDN_PROJECTION_ARITHMETIC',
    'moe_router': 'LLAMINAR_MOE_ROUTER_ARITHMETIC',
    'hc_up': 'LLAMINAR_QFN_HC_UP_ACTIVATION',
    'gdn_output': 'LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION',
    'fa_output': 'LLAMINAR_QFN_FA_OUTPUT_ACTIVATION',
    'gdn_qkv': 'LLAMINAR_QFN_GDN_QKV_ACTIVATION',
}


def file_sha256(path):
    """Hash the actual small executable; never read a model or logit payload."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def original_source_identity(path, descriptor):
    """Authenticate the reference owner's exact stat descriptor without payload reads.

    A current before/after pin alone cannot detect replacement that predates
    admission. Reference generation owns these six immutable metadata fields.
    """
    pin = identity(path)
    observed = dict(path=pin[0],device=pin[1],inode=pin[2],bytes=pin[3],mtime_ns=pin[4],ctime_ns=pin[5])
    try:expected = json.loads(descriptor['object_identity'])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('Original source needs reference-issued stat identity') from error
    require(type(expected) is dict and set(expected) == set(observed) and expected == observed and
            all(type(expected[k]) is type(v) for k,v in observed.items()),
            'Original source no longer matches reference-issued metadata identity')
    return pin


def select_topologies(corpus, selected):
    """Select only reviewed corpus-owned identities, preserving requested order."""
    available = {t['id']: t for t in corpus['manifest']['topologies']}
    names = selected if selected else list(available)
    require(names and len(names) == len(set(names)) and all(n in available for n in names),
            'Unknown or duplicate selected corpus topology')
    return [available[n] for n in names]


def validate_described_config(topology, described):
    """Require the actual public parser's model-free output and exact original argv."""
    argv = topology.get('engine_argv')
    require(type(argv) is list and argv and all(type(v) is str and v and '\0' not in v for v in argv),
            'Corpus topology needs original canonical engine_argv export')
    require(type(described) is dict and described.get('schema') == 'llaminar.model-kld-described-config.v1' and
            described.get('engine_argv') == argv and described.get('config') == topology['expected_config'] and
            described.get('runtime_config_digest') == topology['runtime_config_digest'] == metadata_digest(described['config']),
            'Native public parser differs from source-owned argv/config')
    return argv


def physical_devices(config):
    """Project only original symbolic GlobalDeviceAddress fields for resource observation.

    This does not resolve placement or grant devices. Native compares the entire
    original config and validates its actual admitted execution plan separately.
    """
    found = set()
    def visit(value):
        if type(value) is dict:
            if 'device_type' in value and 'device_ordinal' in value:
                backend = {'CPU':'cpu','CUDA':'cuda','ROCm':'rocm'}.get(value['device_type'])
                ordinal = value['device_ordinal']
                require(backend is not None and type(ordinal) is int and ordinal >= 0,
                        'Invalid explicit physical device in canonical config')
                found.add(backend+':'+str(ordinal))
            for child in value.values(): visit(child)
        elif type(value) is list:
            for child in value: visit(child)
    visit(config)
    require(found, 'Canonical config has no original physical device inventory')
    return sorted(found)


def validate_absolute_pilot(policy, manifest, topology):
    """Validate actual selected row-group roles/limits before native admission.

    The installed evaluator alone parses policy meaning. Expected original rows
    establish groups without inventing candidate metrics or numerical evidence.
    """
    groups = {(topology['backend'], topology['id'], row['context_group'], row['phase'])
              for row in corpus_v2.expected_rows(manifest)}
    _, roles = corpus_v2._absolute_pilot_limits(policy, manifest, groups)
    return dict(policy='initial_absolute_pilot',
        required_groups=[dict(zip(corpus_v2.GROUP_FIELDS,key)) for key in sorted(groups) if roles[key]=='required'],
        diagnostic_groups=[dict(zip(corpus_v2.GROUP_FIELDS,key)) for key in sorted(groups) if roles[key]=='diagnostic'],
        diagnostic_verdict='UNAVAILABLE', whole_corpus_accepted=False)


@dataclass(frozen=True)
class PairedCorpusSelection:
    """One existing exact canonical case and its pinned local document obligations.

    This value neither attaches a profile nor declares topology/certification.
    The source-owned exported case is the only placement/argv authority. Model,
    driver and process ownership remain in the existing CorpusExecution runner.
    """
    exact: str
    record: dict
    corpus: dict
    topology: dict
    paths: dict
    pins: dict
    documents: dict
    canonical_arguments: list

    @classmethod
    def from_export(cls, exact, record, corpus_root, document_root):
        """Authenticate all declared small documents before selecting or launching a model."""
        require(type(record) is dict and record.get('model_parity_schema') == 1 and
                type(exact) is str and '.ProductionParity/' in exact,
                'Paired acceptance needs one original registered ProductionParity cell')
        pair = record.get('paired_workload')
        require(type(pair) is dict and set(pair) == {'schema','policy','certification_eligible','role',
                'case_id','source_fields','accuracy','benchmark','arithmetic'} and type(pair['schema']) is int and pair['schema']==2 and
                pair['policy']=='diagnostic_obligation' and pair['certification_eligible'] is False and
                pair['role'] in ('required','expanded') and pair['case_id']==record.get('id'),
                'Missing or malformed source-owned paired workload; no automatic profile attachment')
        fields = ['model'] + (['shared_predictor'] if 'shared_predictor' in record else [])
        require(pair['source_fields']==fields and all(type(record.get(k)) is str and record[k] for k in fields),
                'Paired source roles differ from original canonical cell')
        require(record.get('runtime',{}).get('generation',{}).get('mtp_policy')=='off',
                'Local QFN paired acceptance supports ordinary MTP-off cells only')
        accuracy, benchmark = pair['accuracy'], pair['benchmark']
        common = {'arguments_from','catalog','workload','policy'}
        require(type(accuracy) is dict and accuracy.get('arguments_from')=='diagnostic_benchmark' and
                type(benchmark) is dict and set(benchmark)=={'policy','arguments_from','profile'} and
                benchmark['policy']=='diagnostic_declared' and benchmark['arguments_from']=='diagnostic_benchmark',
                'Paired obligations must reuse the original diagnostic argv')
        policy = accuracy.get('policy')
        if policy=='initial_absolute_pilot':
            require(set(accuracy)==common|{'pilot','approved_baseline','calibrated','whole_corpus_accepted'} and
                    all(accuracy[k] is False for k in ('approved_baseline','calibrated','whole_corpus_accepted')),
                    'Initial paired pilot cannot claim baseline/calibration/whole-corpus acceptance')
            names = ('pilot',)
        else:
            require(policy=='calibrated_revision' and set(accuracy)==common|{'baseline','approval','calibration'},
                    'Paired revision needs actual baseline, approval and calibration documents')
            names = ('baseline','approval','calibration')
        refs = {k:accuracy[k] for k in ('catalog','workload',*names)}
        refs['benchmark'] = benchmark['profile']
        paths, pins, documents = {}, {}, {}
        for name, reference in refs.items():
            require(type(reference) is dict and set(reference)=={'path','sha256'} and
                    type(reference['path']) is str and reference['path'] and '\0' not in reference['path'] and
                    type(reference['sha256']) is str and len(reference['sha256'])==64 and
                    all(c in '0123456789abcdef' for c in reference['sha256']),
                    'Paired document needs its exact source-owned path and SHA256')
            path = (Path(document_root)/reference['path']).resolve(strict=True)
            pin = identity(path)
            require(path.is_file() and 0 < pin[3] <= 16*1024*1024, 'Paired document exceeds bounded metadata extent')
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest()==reference['sha256'] and identity(path)==pin,
                    'Paired document changed or has a foreign digest: '+name)
            document = json.loads(raw)
            require(type(document) is dict, 'Paired document must be a JSON object: '+name)
            paths[name], pins[str(path)], documents[name] = path, pin, document
        root = Path(corpus_root).resolve(strict=True)
        require(paths['workload']==(root/'manifest.json').resolve(strict=True),
                'Paired workload must be the exact approved corpus manifest, not a searched generation')
        corpus = corpus_v2.load_corpus(root, catalog=paths['catalog'])
        require(corpus['manifest']==documents['workload'], 'Paired corpus differs from its declared workload')
        matches = [t for t in corpus['manifest']['topologies'] if t['canonical_cell']==exact]
        require(len(matches)==1, 'Paired corpus needs exactly one topology for the original exact cell')
        topology = matches[0]
        require(type(pair['arithmetic']) is dict and set(pair['arithmetic'])==set(SELECTORS) and
                pair['arithmetic']==topology['arithmetic'],
                'Corpus arithmetic differs from explicit original paired-profile selectors')
        source = corpus['manifest']['source']
        require(Path(record['model']).resolve(strict=True)==Path(source['shards'][0]['path']).resolve(strict=True),
                'Paired corpus names another original main source')
        require(('shared_predictor' in record)==('shared_predictor' in source),
                'Paired corpus has a missing or extra original predictor source role')
        if 'shared_predictor' in record:
            require('shared_predictor' in source and Path(record['shared_predictor']).resolve(strict=True)==
                    Path(source['shared_predictor']['path']).resolve(strict=True),
                    'Paired corpus names another original predictor source')
        if policy=='initial_absolute_pilot':
            validate_absolute_pilot(documents['pilot'], corpus['manifest'], topology)
        else:
            # Use the existing evaluator's row and calibration validators;
            # document authentication never fabricates a candidate/self-baseline.
            baseline, approval = documents['baseline'], documents['approval']
            corpus_v2.validate_observation(corpus['manifest'],baseline)
            require(baseline['topology_id']==topology['id'] and baseline['backend']==topology['backend'] and
                    approval.get('schema')=='llaminar.model-kld-baseline-approval.v1' and
                    approval.get('corpus_manifest_digest')==metadata_digest(corpus['manifest']) and
                    approval.get('baseline_report_digest')==metadata_digest(baseline),
                    'Paired revision is not bound to its actual approved baseline/topology')
            groups={(topology['backend'],topology['id'],row['context_group'],row['phase'])
                    for row in corpus_v2.expected_rows(corpus['manifest'])}
            corpus_v2._limits(documents['calibration'],corpus['manifest'],baseline,approval,groups)
        profile = record.get('diagnostic_benchmark',{})
        require(type(profile) is dict, 'Paired case lacks its original typed diagnostic profile')
        arguments = profile.get('args')
        require(profile.get('schema') == 1 and type(profile.get('schema')) is int and
                profile.get('policy') == 'diagnostic_declared' and
                profile.get('certification_eligible') is False and
                type(profile.get('context_length')) is int and profile['context_length'] > 0 and
                type(arguments) is list and arguments and
                all(type(v) is str and v and '\0' not in v for v in arguments),
                'Paired case lacks its original typed diagnostic profile')
        # The exporter separates policy arguments from the model and context,
        # just as the benchmark launcher does. Bind those source-owned fields
        # before asking the public parser to authenticate the full request.
        canonical_arguments = [*arguments, '-m', record['model'],
                               '--context-length', str(profile['context_length'])]
        unchanged(pins)
        return cls(exact, json.loads(json.dumps(record)), corpus, topology, paths, pins, documents, canonical_arguments)

    def require_public_projection(self, runner, args, environment, observed):
        """Compare actual parser-issued intent; never interpret topology/CLI in Python."""
        unchanged(self.pins)
        validate_described_config(self.topology, observed)
        canonical = json.loads(runner.subprocess.check_output(
            [str(args.binary),'--describe-config','--',*self.canonical_arguments],env=environment,text=True))
        require(canonical.get('schema')=='llaminar.model-kld-described-config.v1' and
                canonical.get('engine_argv')==self.canonical_arguments and
                canonical.get('runtime_config_digest')==metadata_digest(canonical.get('config')),
                'Canonical public parser did not preserve original exported arguments')
        for described in (canonical, observed):
            require(type(described.get('qfn_canonical_intent')) is dict and described['qfn_canonical_intent'] and
                    described.get('qfn_arithmetic_intent')==self.topology['arithmetic'],
                    'Paired parser omitted typed QFN placement/math or changed actual arithmetic')
        require(canonical['qfn_canonical_intent']==observed['qfn_canonical_intent'],
                'Paired observer changed original canonical placement/math intent')
        config = canonical['config']['configuration']
        for role, field in (('model','model_path'),('shared_predictor','shared_predictor')):
            expected = self.record.get(role)
            actual = config.get(field)
            require((expected is None and actual is None) or (expected is not None and type(actual) is str and
                    Path(expected).resolve(strict=True)==Path(actual).resolve(strict=True)),
                    'Canonical parser changed original source role: '+role)
        unchanged(self.pins)
        return canonical

    def observation(self):
        """Retain original obligations separately from numerical and benchmark outcomes."""
        return dict(exact_cell=self.exact, source_export=self.record,
            documents={name:dict(path=str(path),sha256=(self.record['paired_workload']['benchmark']['profile'] if name=='benchmark' else
                self.record['paired_workload']['accuracy'][name])['sha256']) for name,path in self.paths.items()},
            benchmark_status='DECLARED_NOT_EXECUTED',certification_eligible=False)


class CorpusExecution:
    """One immutable selected corpus topology; no second execution lifecycle."""

    def __init__(self, corpus, topology, *, baseline=None, approval=None, calibration=None, absolute_pilot=None, paired=None):
        """Pin original sources and optional reviewed numerical inputs before admission."""
        self.corpus = corpus; self.topology = topology; self.paired = paired
        self.baseline = baseline; self.approval = approval; self.calibration = calibration
        self.absolute_pilot = absolute_pilot
        require(absolute_pilot is None or not any(value is not None for value in (baseline,approval,calibration)),
                'Absolute pilot cannot use a baseline, approval or calibration')
        require((baseline is None) == (approval is None), 'Baseline report and approval must be supplied together')
        require(baseline is not None or calibration is None, 'Calibration requires an approved baseline report')
        self.pins = dict(corpus['inputs'])
        if paired is not None:self.pins.update(paired.pins)
        self.sources = []
        source = corpus['manifest']['source']
        for descriptor in source['shards'] + ([source['shared_predictor']] if 'shared_predictor' in source else []):
            path = Path(descriptor['path']).resolve(strict=True)
            pin = original_source_identity(path,descriptor)
            require(path.name == descriptor['name'] and pin[3] == descriptor['bytes'], 'Original source extent/name changed')
            self.pins[str(path)] = pin; self.sources.append(path)
        self.devices = physical_devices(topology['expected_config'])
        self.backends = sorted({d.split(':')[0] for d in self.devices})
        self.environment = {SELECTORS[k]:topology['arithmetic'][k] for k in SELECTORS}
        for key in ('baseline', 'approval', 'calibration', 'absolute_pilot'):
            path = getattr(self, key)
            if path is not None:
                path = Path(path).resolve(strict=True); self.pins[str(path)] = identity(path)
                setattr(self, key, json.loads(path.read_text()))
        if self.absolute_pilot is not None:
            validate_absolute_pilot(self.absolute_pilot, corpus['manifest'], topology)
        self.binary_sha256 = None

    def prepare(self, runner, args, output, env):
        """Publish a native descriptor and authenticate parser output before model work."""
        unchanged(self.pins)
        argv = self.topology.get('engine_argv')
        # Validate argv shape before passing it to a subprocess. The actual
        # parser remains the sole config meaning/roundtrip authority.
        require(type(argv) is list and argv and all(type(v) is str and v and '\0' not in v for v in argv),
                'Missing source-owned canonical engine argv')
        self.binary_pin = identity(args.binary)
        self.binary_sha256 = file_sha256(args.binary)
        described = json.loads(runner.subprocess.check_output(
            [str(args.binary),'--describe-config','--',*argv], env=env, text=True))
        validate_described_config(self.topology, described)
        runner.write_json(output/'described-config.json', described)
        if self.paired is not None:
            canonical = self.paired.require_public_projection(runner,args,env,described)
            runner.write_json(output/'canonical-described-config.json',canonical)
            self.pins[str(output/'canonical-described-config.json')]=identity(output/'canonical-described-config.json')
        workload = output/'workload.json'
        runner.write_json(workload, dict(schema='llaminar.model-kld-workload.v2',
            corpus=self.corpus['manifest'], topology_id=self.topology['id']))
        self.pins[str(workload)] = identity(workload)
        self.pins[str(output/'described-config.json')] = identity(output/'described-config.json')
        return workload, argv

    def score(self, runner, payload, output):
        """Join real native proof to actual binary and publish uncalibrated or approved metrics."""
        unchanged(self.pins)
        require(identity(Path(self.binary_pin[0])) == self.binary_pin, 'Native observer executable changed')
        raw = json.loads((payload/'logits.json').read_text())
        perf = json.loads((output/'perf-stats.json').read_text())
        require(type(perf.get('records')) is list and perf['records'], 'Missing original native PerfStats artifact')
        require(raw.get('native_identity',{}).get('executable_sha256') == self.binary_sha256,
                'Native observer executable proof differs from launched binary')
        require(raw.get('actual_owner') and type(raw.get('path_records')) is list,
                'Missing actual original owner/path evidence')
        metrics = corpus_v2.score_observation(self.corpus,payload/'logits.json')
        runner.write_json(output/'metrics.json', metrics)
        summary = corpus_v2.summarize(self.corpus['manifest'],metrics,self.baseline,self.approval,calibration=self.calibration,absolute_pilot=self.absolute_pilot)
        runner.write_json(output/'numerical-summary.json', summary)
        unchanged(self.pins)
        result = dict(passed=summary['passed'],verdict=summary['verdict'],metrics=metrics,
            numerical_summary=summary,observations_complete=True,production_path=raw['proofs'],
            native_identity=raw['native_identity'],actual_owner=raw['actual_owner'],path_records=raw['path_records'],
            reset_byte_equal=True,driver_window_clean=True)
        if self.absolute_pilot is not None:
            result.update(scope=summary['scope'],acceptance_scope=summary['acceptance_scope'],
                calibrated=False,certification_eligible=False,whole_corpus_accepted=False,
                diagnostic_verdict='UNAVAILABLE',absolute_pilot_digest=summary['absolute_pilot_digest'])
        return result
