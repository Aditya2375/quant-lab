/* Cross-checks docs/assets/engine.js against quadrature-verified references
   and statistical tolerances. Run: node tests/test_engine_js.mjs */
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const E = require('../docs/assets/engine.js');

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`ok   ${name}`);
  else { failures++; console.log(`FAIL ${name} ${detail || ''}`); }
}

const REF_CALL = 5.349804578880892; // scipy quadrature, +/-1.5e-12
const p = E.bsPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call');
check('bs matches quadrature reference', Math.abs(p - REF_CALL) < 1e-9, p);

const put = E.bsPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'put');
check('put-call parity', Math.abs((p - put) - (100 * Math.exp(-0.005) - 105 * Math.exp(-0.015))) < 1e-10);

check('cdf machine precision', Math.abs(E.normCdf(1.96) - 0.9750021048517795) < 1e-12);
check('cdf tail', Math.abs(E.normCdf(-6) - 9.865876450377e-10) / 9.865876450377e-10 < 1e-6);
check('ppf roundtrip', Math.abs(E.normCdf(E.normPpf(1e-6)) - 1e-6) < 1e-10);

check('iv roundtrip', Math.abs(E.impliedVol(p, 100, 105, 0.5, 0.03, 0.01, 'call') - 0.25) < 1e-9);

const g = E.bsGreeks(100, 105, 0.5, 0.03, 0.25, 0.01, 'call');
const h = 1e-5;
const fd = (E.bsPrice(100 + h, 105, 0.5, 0.03, 0.25, 0.01, 'call') - E.bsPrice(100 - h, 105, 0.5, 0.03, 0.25, 0.01, 'call')) / (2 * h);
check('delta finite diff', Math.abs(g.delta - fd) < 1e-7);

const mc = E.mcPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call', 200000, 42, false, false);
check('mc within 4 se', Math.abs(mc.price - REF_CALL) < 4 * mc.se, `${mc.price} se=${mc.se}`);

const av = E.mcPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call', 50000, 1, true, true);
const pl = E.mcPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call', 50000, 1, false, false);
check('variance reduction not worse', av.se <= pl.se);

const sob = E.qmcPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call', 65536, 5);
check('qmc beats plain mc', Math.abs(sob.price - REF_CALL) < Math.abs(pl.price - REF_CALL));
check('qmc accurate', Math.abs(sob.price - REF_CALL) < 1e-3, sob.price);

const det1 = E.mcPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call', 10000, 9, true, true);
const det2 = E.mcPrice(100, 105, 0.5, 0.03, 0.25, 0.01, 'call', 10000, 9, true, true);
check('deterministic', det1.price === det2.price);

if (failures) { console.error(failures + ' failures'); process.exit(1); }
console.log('all js engine checks passed');
