/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fcp/client/attestation/extract_attestation_records.h"

#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "fcp/client/attestation/log_attestation_records.h"
#include "fcp/client/attestation/test_values.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/protos/confidentialcompute/verification_record.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::client::attestation {
namespace {
using ::fcp::client::attestation::test_values::GetKnownValidEncryptionConfig;
using ::testing::IsEmpty;
using ::testing::SizeIs;

// A test LogSink that captures all logs in text form into a stringstream buffer
// for later inspection.
class TestLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    stream_ << entry.text_message_with_prefix_and_newline();
  }

  // Returns the istream from which captured logs can be read.
  std::istream& stream() { return stream_; }

 private:
  std::stringstream stream_;
};

// Tests that the record extractor successfully handles the original output
// format, which previous incorrectly output more than 200 chars at a time when
// run on Android, and that it will continue to do so in the future.
TEST(ExtractAttestationRecordsTest, ExtractRecordWithFormatV1) {
  // Note that this log stream contains entries that:
  // a) are longer than 200 chars,
  // b) don't always end with a closing bracket,
  // c) have content that overlaps with the next line.
  std::stringstream hardcoded_record(R"text(
I0424 09:57:22.979757    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:57:22.979822    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <H4sIAAAAAAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9QycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu/LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDpz+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZCIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx8aQft9RcUx6MC7+iQ9Icxu
I0424 09:57:22.979842    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <ycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu/LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDpz+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZCIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUionKjV0AQz5Hd5REPDZsj29F
I0424 09:57:22.979853    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: </LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDpz+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZCIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUionKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPzWtA18O2+4j3W2Or3ekVtK2
I0424 09:57:22.979861    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <z+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZCIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUionKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPzWtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGuT3OOec7dUWPyoUOnhiZ6Sf
I0424 09:57:22.979871    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <CIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUionKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPzWtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGuT3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLXbuj0Y9bTpHO6UTZ1ueuIhq
I0424 09:57:22.979877    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUionKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPzWtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGuT3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLXbuj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaWg1P5eFAtAoCDlxN1HFJWpI
I0424 09:57:22.979883    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <nKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPzWtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGuT3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLXbuj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaWg1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogiUDFsZUqm0WEERqgcOp3BDD
I0424 09:57:22.979887    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <WtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGuT3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLXbuj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaWg1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogiUDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd4LQhmfveQjlgCssBI/XpOo
I0424 09:57:22.979892    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <T3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLXbuj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaWg1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogiUDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLfCcR91ay9DRkxCJmYP/aEGz
I0424 09:57:22.979896    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <buj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaWg1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogiUDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLfCcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU5P7zrJ794GmEyzegNvqtDv
I0424 09:57:22.979899    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <g1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogiUDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLfCcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98/1CbpUKQncNuRaXK93FxTT
I0424 09:57:22.979914    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <UDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLfCcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98/1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2NpymS80ezRQcpui4V+P6uZ6J2S
I0424 09:57:22.979918    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLfCcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98/1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2NpymS80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:57:22.979922    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <CcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98/1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2NpymS80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:57:22.979925    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98/1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2NpymS80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:57:22.979930    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: </1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2NpymS80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:57:22.979934    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <S80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:57:22.979937    9092 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
  )text");

  // Extract the serialized log record from the stream.
  std::vector<Record> records;
  ExtractRecords(hardcoded_record,
                 [&](Record record) { records.push_back(record); });
  ASSERT_THAT(records, SizeIs(1));
  EXPECT_EQ(records[0].start_line, 3);
  EXPECT_EQ(records[0].end_line, 20);
}

// Tests that the record extractor successfully handles the current output
// format (and will continue to do so in the future).
TEST(ExtractAttestationRecordsTest, ExtractRecordSuccessfully) {
  // Create a verification record with a good amount of data in it.
  confidentialcompute::AttestationVerificationRecord record;
  auto encryption_config = GetKnownValidEncryptionConfig();
  *record.mutable_encryption_key() = encryption_config.encryption_key();
  *record.mutable_pipeline_configuration() =
      test_values::GetKnownValidSignedEndorsements();

  // Call the LogSerializedVerificationRecord function and gather the log
  // records it emits.
  TestLogSink log_sink;
  absl::AddLogSink(&log_sink);
  LogSerializedVerificationRecord(record);
  absl::RemoveLogSink(&log_sink);

  // Extract the serialized log record from the stream.
  std::vector<Record> records;
  int num_records = ExtractRecords(
      log_sink.stream(), [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 1);
  ASSERT_THAT(records, SizeIs(1));

  // Verify that the extracted record matches the original record.
  fcp::confidentialcompute::AttestationVerificationRecord parsed_record;
  ASSERT_TRUE(parsed_record.ParseFromString(records[0].decompressed_record));
  EXPECT_THAT(parsed_record, EqualsProto(record));

  // Now also compare the original record to the hardcoded serialized record
  // below. It should match. If this test starts failing this indicates that the
  // serialization format may have changed, and in that case this hardcoded
  // record should be moved into its own test (like `ExtractRecordsWithFormatV1`
  // above), and a new hardcoded record should be generated and added below.
  //
  // This will help ensure that the record extraction utility continues to
  // properly parse both records in previously used formats as well as records
  // in whatever new format we may come up with.
  //
  // The sample log stream below was captured from an Android client executing
  // the confidential aggregation protocol with attestation transparency
  // verification enabled.
  std::stringstream hardcoded_record(R"text(
08-26 01:56:36.136 24625 24648 D fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
08-26 01:56:36.136 24625 24648 D fcp.attest: <H4sIAAAAAAAAA4VVeVRTZxYnGDCGpZyIG4zyeNSFsOS9QBIDKiSQEANBCVkIMtaX9x5J4GUxC8hWkIp1LRYRtE51RivWqh1QREAqIq2ABQ5FBFzABeuGUwHHykiBCdr2nPlr/vjuPb/7+7577+9859wLjrpTIQrJg+RF6TjdXLgVqjwEAoYAjJ+gHV8wEkN6UtRxGZX/>
08-26 01:56:36.136 24625 24648 D fcp.attest: <s7QFHr1X7Fl2tHh42CeG1uBO5dFcKNt7anucPa5cbsW9nCk7p8/3OIN2/2yyx5kPzGT6bC/nWZnH0ZKYfQ8u0BuMPl+4NIWva2c7dI34TVIcaZGjwGjv29WiEV9x7vQreXdf0K1LS1QC9gC58Z56vL/8vr9VemGpo6rVt3d71/PA2OLMBS2rdtC+2rF7QBHFjasKole7>
08-26 01:56:36.136 24625 24648 D fcp.attest: <USUURw+yF7l46eWvQR5ZmUNuCqrVthW8rnu5Wu05xXvg8H1bxPHHaMum/JbGbTsmPe9dKB1L2jwYcLszp5NHrXb+4trZvdZ64KBP4Kt/lV6nPXSl3iHZe3OllA0W2sX1lXY/Inm5Uh7e/ROBdq61/Q9Ej9JarSZLGIOh0Vm1NnUwatQzTGZjGo5ag4xIOmPmqAmjmqFH>
08-26 01:56:36.136 24625 24648 D fcp.attest: <dAYGZkQtDKuZgRKITs/gMrkh3GA9xtRSP6Q4hgFOgYuzbtEMi7jfKF5eKzudi511gfNnd6Bf3jnOkRE1/6BFNlbvvLLrhPXJxhWl30ouVUyoOU8HP5bhF7z9DTDFo2ptgbsqz1Z+TgE73817o/GjiGSR01mV50bPEY7NA3uDxrwmqNRxKvUsOQdETDoFbrbojAYwDISC>
08-26 01:56:36.136 24625 24648 D fcp.attest: <oWAYDATTdQbMDrWIRYtjZjzdaMbsUYsJR8GwHBBDrMiMn6FnPEJojGa7bL39iUWLMFls++UMhLDh9gDODQ1BmSjOTuWEIkymOlSNpbIgZggEq3E1BuHq1JWhUCoXRzDWSpjDYqKoGsY5cCgLZaaibJgF5uXZK+s0BsRqM+Mz5VCjwYobrPbcEoE8ai1PreVAclMIS8E1>
08-26 01:56:36.136 24625 24648 D fcp.attest: <xWBslSZRlW3RCDgqfnomZDSj8SF8gZXgyWWbEZ5OwLOJVDEGqRJDDeIsqSjVoJQphZkyycqApNQMjZSVpM9gcMWalZp1W0JX22WYbGpCh8biWf9bOS4RssYlwlFSOSaW6fgJCrlYkiiXaBIhRfJ7DoqVyZPNGKQIUcq1cWrYJMb1xswESChVyok/sVQglKogqTQZEgqV>
08-26 01:56:36.136 24625 24648 D fcp.attest: <aVo2CmMGtT7eHJ+ESZJFhCwhO56fKDQppcxkkYIpDEV4JnMiQWxRJwl1qEyYiSZJsxMNmEyll8NqgcmWLJRk4gZthpTgb1HL4UzMoDDJ5FtgpRLTJYp4sCKaIPBoIlQGEyxpNsFfL4Nif9cilKUnaORCRZRMQESvFViFyvc6rFGa1avtv5CX53H+wu0hysIXNXYLAgO/>
08-26 01:56:36.136 24625 24648 D fcp.attest: <zhamFEirp44NdgqOVCRtuon4i4e3r6t8OrRsFv3FHhC4NnDUq6KI49ZY4XkWee79qx/qkRHSURUB3Ck7rWn+qRsEBPWPnOGm5mc5Cxh0YzXYsf9Ydf2azz7oTjCtWDLlXZQLAh+kDM1qjSm+evDUBlLxns7ZtqKPt53YkDBNO3jv0q0loBMILD5c1HTq4+V/3eCxyPQR>
08-26 01:56:36.136 24625 24648 D fcp.attest: <tcJwfCotkte2+y/7893Lmcmmb0BgvDhHfjSXefu7YiiszrzMX5j7trb5bd0C/bx281nMmAACm8ibLxZRlVF3rvpE/NxwZNmpwQL3j56/fnmkhOzoICZ2gsD5WDc2wzKVP52RN3nLe1627fqzvtas6MQzgVOFoxHu/SBwc/BBzewho6/f9Lb2cp83e3SDFPFgTwn15vXe>
08-26 01:56:36.136 24625 24648 D fcp.attest: <mvaKQ89BAO0a7s4mCkbWhfv+py98YUOKpf7xmPgx7cuQIVV06uM2EMgZXOp2eSLF7bjygWrVGTRmn8Y76nxKn8jNP+Ja+2mLGATuzh+uCgn/tOHMqQPkzZXTJ/1ujHQK+G8ckg/4PuZVkMdAwCbegWQ03SzJdQpf//fB9pX5tqZ2vHfWbwE9aVg93jUPBDTby+lyx+FD>
08-26 01:56:36.136 24625 24648 D fcp.attest: <Rz0nt8X+pGr8pG3rZqLW737MIteDO9t/SLOrFVw6PPvMMSxtqd+EVBKZLUh6+lr7S9/DAxPlbumd6x+BADvC0aPOcX6h7EpR/MjJjB/cecTcRynDUx/OKSts41U3gEDlVpK3Ivmk+mBwr/e3b87zyXiXK0dbHrs+UuR29YTnDRB4VeQkqXRe8VVNksCTd9iXk/LdlsmF>
08-26 01:56:36.136 24625 24648 D fcp.attest: <N0jITerAiPOPqwAQ6Hng2HV8yedpr0n1IreifsrPr/Irdo3NdwiQyO7RU7sDQOA6+dqNH5v6ecuWzuoTjmb5xO8/cDK8f467a5T4xTh/TRwIfJ/fTG3ZvaRxI3373LHT7oKSSx3K6EVFn9ZtIBSVk0+c6SveDbhg+4yxWI1mPBjDM4AgAIa5IRAL4rK4XJjNZrFZEDss>
08-26 01:56:36.136 24625 24648 D fcp.attest: <crkecSp1yZzb/aimKVh0JrRuMXfew7b+6bXHxkpc7od+MnHx0KbdKbW5cGvGwMsDbh5ea6PXx2gFFFOs4GXz0N+8+WQHBwcSOGMp9EMkKkj9vxuSZqH62zeRC2XXuzVbWbgvzsuF0tv9BwDtzLH9/e8BLbLciU0fj10W5kRCxGBg2kj6xt+oDi2dJJXlbpV80W3p58sX>
08-26 01:56:36.136 24625 24648 D fcp.attest: <f70JvriGMv9Kn5bG7ns7XFslqi/4pV1xbnHLv5E573pz/S8Nf64pEAgAAA==>
08-26 01:56:36.136 24625 24648 D fcp.attest: <>
  )text");

  // Extract the serialized log record from the hardcoded logs above.
  records.clear();
  num_records = ExtractRecords(
      hardcoded_record, [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 1);
  ASSERT_THAT(records, SizeIs(1));
  EXPECT_EQ(records[0].start_line, 3);
  EXPECT_EQ(records[0].end_line, 17);

  // Verify that the extracted record from the hardcoded logs matches the
  // original record.
  ASSERT_TRUE(parsed_record.ParseFromString(records[0].decompressed_record));
  EXPECT_THAT(parsed_record, EqualsProto(record));
}

// Tests that the record extractor can successfully extract multiple sequential
// records.
TEST(ExtractAttestationRecordsTest, ExtractRecordMultipleSuccessfully) {
  // Create a verification record with a good amount of data in it.
  confidentialcompute::AttestationVerificationRecord record1;
  auto encryption_config = GetKnownValidEncryptionConfig();
  *record1.mutable_encryption_key() = encryption_config.encryption_key();
  *record1.mutable_pipeline_configuration() =
      test_values::GetKnownValidSignedEndorsements();

  confidentialcompute::AttestationVerificationRecord record2 = record1;
  record2.mutable_pipeline_configuration()->set_payload("another tag");

  // Call the LogSerializedVerificationRecord function twice and gather the log
  // records it emits.
  TestLogSink log_sink;
  absl::AddLogSink(&log_sink);
  LogSerializedVerificationRecord(record1);
  LogSerializedVerificationRecord(record2);
  absl::RemoveLogSink(&log_sink);

  // Extract the serialized log records from the stream. There should be two.
  std::vector<Record> records;
  int num_records = ExtractRecords(
      log_sink.stream(), [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 2);
  ASSERT_THAT(records, SizeIs(2));
  EXPECT_NE(records[0].decompressed_record, records[1].decompressed_record);

  // Verify that the extracted records match the original records.
  fcp::confidentialcompute::AttestationVerificationRecord parsed_record;
  ASSERT_TRUE(parsed_record.ParseFromString(records[0].decompressed_record));
  EXPECT_THAT(parsed_record, EqualsProto(record1));
  ASSERT_TRUE(parsed_record.ParseFromString(records[1].decompressed_record));
  EXPECT_THAT(parsed_record, EqualsProto(record2));
}

// Tests that the utility ignores truncated records and continues extracting any
// other valid records that may still exist in the stream.
TEST(ExtractAttestationRecordsTest,
     IgnoreTruncatedRecordsAndInterleavedTextAndContinueExtracting) {
  std::stringstream hardcoded_record(R"text(
--- THIS IS A TRUNCATED RECORD (WE'VE REMOVED SOME OF ITS LINES), AND SHOULD BE IGNORED.
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <H4sIAAAAAAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9Q>
I0424 09:51:46.084895    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <ycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu>
I0424 09:51:46.084900    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDp>
< SNIP! >
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
--- THIS IS A VALID RECORD WHICH SHOULD STILL BE EXTRACTED.
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <H4sIAAAAAAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9Q>
I0424 09:51:46.084895    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <ycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu>
I0424 09:51:46.084900    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDp>
I0424 09:51:46.084903    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <z+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZ>
I0424 09:51:46.084906    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <CIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx>
I0424 09:51:46.084912    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUio>
I0424 09:51:46.084915    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <nKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPz>
I0424 09:51:46.084918    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <WtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGu>
I0424 09:51:46.084921    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <T3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLX>
I0424 09:51:46.084923    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <buj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaW>
I0424 09:51:46.084926    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <g1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogi>
I0424 09:51:46.084930    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <UDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd>
I0424 09:51:46.084933    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLf>
I0424 09:51:46.084936    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <CcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU>
I0424 09:51:46.084939    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98>
I0424 09:51:46.084941    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2Npym>
I0424 09:51:46.084947    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <S80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
  )text");

  // Extract the serialized log record from the hardcoded logs above.
  std::vector<Record> records;
  int num_records = ExtractRecords(
      hardcoded_record, [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 1);
  ASSERT_THAT(records, SizeIs(1));
  EXPECT_EQ(records[0].start_line, 11);
  EXPECT_EQ(records[0].end_line, 28);
}

// Tests that the utility ignores records with non-base64 data and continues
// extracting any other valid records that may still exist in the stream.
TEST(ExtractAttestationRecordsTest,
     IgnoreInvalidBase64RecordsAndInterleavedTextAndContinueExtracting) {
  std::stringstream hardcoded_record(R"text(
--- THIS IS A RECORD WITH NON-BASE64 CONTENT (IT HAS '%' CHARS), AND SHOULD BE IGNORED.
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <%%%%%%%%AAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9Q>
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
--- THIS IS A VALID RECORD WHICH SHOULD STILL BE EXTRACTED.
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <H4sIAAAAAAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9Q>
I0424 09:51:46.084895    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <ycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu>
I0424 09:51:46.084900    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDp>
I0424 09:51:46.084903    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <z+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZ>
I0424 09:51:46.084906    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <CIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx>
I0424 09:51:46.084912    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUio>
I0424 09:51:46.084915    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <nKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPz>
I0424 09:51:46.084918    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <WtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGu>
I0424 09:51:46.084921    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <T3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLX>
I0424 09:51:46.084923    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <buj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaW>
I0424 09:51:46.084926    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <g1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogi>
I0424 09:51:46.084930    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <UDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd>
I0424 09:51:46.084933    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLf>
I0424 09:51:46.084936    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <CcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU>
I0424 09:51:46.084939    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98>
I0424 09:51:46.084941    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2Npym>
I0424 09:51:46.084947    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <S80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
  )text");

  // Extract the serialized log record from the hardcoded logs above.
  std::vector<Record> records;
  int num_records = ExtractRecords(
      hardcoded_record, [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 1);
  ASSERT_THAT(records, SizeIs(1));
  EXPECT_EQ(records[0].start_line, 8);
  EXPECT_EQ(records[0].end_line, 25);
}

// Tests that the utility ignores records with invalid compressed data (validly
// base64, but not validly gzipped data after base64 decoding) and continues
// extracting any other valid records that may still exist in the stream.
TEST(ExtractAttestationRecordsTest,
     IgnoreInvalidGzippedRecordsAndInterleavedTextAndContinueExtracting) {
  std::stringstream hardcoded_record(R"text(
--- THIS IS A RECORD WITH INVALID GZIPPED DATA (WE'VE REMOVED THE FIRST ENCODED CHARACTERS), AND SHOULD BE IGNORED.
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCL>
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
--- THIS IS A VALID RECORD WHICH SHOULD STILL BE EXTRACTED.
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <H4sIAAAAAAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9Q>
I0424 09:51:46.084895    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <ycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu>
I0424 09:51:46.084900    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDp>
I0424 09:51:46.084903    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <z+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZ>
I0424 09:51:46.084906    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <CIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx>
I0424 09:51:46.084912    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUio>
I0424 09:51:46.084915    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <nKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPz>
I0424 09:51:46.084918    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <WtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGu>
I0424 09:51:46.084921    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <T3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLX>
I0424 09:51:46.084923    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <buj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaW>
I0424 09:51:46.084926    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <g1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogi>
I0424 09:51:46.084930    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <UDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd>
I0424 09:51:46.084933    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLf>
I0424 09:51:46.084936    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <CcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU>
I0424 09:51:46.084939    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98>
I0424 09:51:46.084941    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: </1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2Npym>
I0424 09:51:46.084947    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <S80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.attest: <>
  )text");

  // Extract the serialized log record from the hardcoded logs above.
  std::vector<Record> records;
  int num_records = ExtractRecords(
      hardcoded_record, [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 1);
  ASSERT_THAT(records, SizeIs(1));
  EXPECT_EQ(records[0].start_line, 8);
  EXPECT_EQ(records[0].end_line, 25);
}

// Tests that the utility ignores records that don't start with "fcp.attest",
// even if they otherwise contain valid record data.
TEST(ExtractAttestationRecordsTest, IgnoreDataNotTaggedWithFcpAttest) {
  std::stringstream hardcoded_record(R"text(
I0424 09:51:46.084833    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: This device is contributing data via the confidential aggregation protocol. The attestation verification record follows.
I0424 09:51:46.084886    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <H4sIAAAAAAAAA+1We1QTVx7OTEIIA1GeorwMWh6C4CSTJ4JAAFetUuVlhFPtJBMwAlFCQALWSkB8oC5aUAR5BYRgoWqxCLsqghalKpSCoqGgBSnKw1ZB3tIm9bC69bh2z3Z79o/9nTN3zv3ud3/39/vmu+cMpDCGDkIkwDBfB8TNBB73ewP4B9u4GfjNmitq+LE7Pu9Q>
I0424 09:51:46.084895    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <ycTAohFBW0vF38KD9vh+IvDzF1IDb8qn3pX7XMLlZ2Hukoxuua7BuhKfIXvdfVWp1ZwX/aMW2z6zwS16Unz/XMlJzD2l/QpSW7dy0e8u+r8USbOeZXyWp7zo8LHhsvvcFGLMyEDW4LTlrNT6rscpRV5Lfn5HzOj4tvzdegucL/HvQ1BU0u7gPvzdWw/3rmUfAC/U99pu>
I0424 09:51:46.084900    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: </LTbft/d3Ds/QWLZkgdhsqozAx6+y3df9vQOX5uQNxZ4e1mgNnUmvz4LwGmed533vxZPELkbef2HBrkdXUez4Z+DnirssnGGTOeYEGfHKxNjksA882qVlfhR7P5NETGjO86/Lc9y5VUfSg4lUYndXE31XznaLu1omJtlnZ5nGXtxH2f3caOqjij9I0UrrioWu/zg7nDp>
I0424 09:51:46.084903    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <z+zx//H2MAstAUAwQOGDqBRuGTY/lV+/IZPB0S3TARDelpAIUgBrHmVLb1wTacv2/aWHMaOwEfOckZ99PbcluhBUuKkYm+T0mgU8SvuXqe97fPfwp679q0BXk4hPFWnbizhHr5h0u/g3L7XkfGh4mwB9S0j2KgBsCwh+ntGyyEihVCIS+Hh5+3vSGMwNwNBJIM4epdEZ>
I0424 09:51:46.084906    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <CIKhKA3BOAwaIxRGMQFbwGdxWBwEZiEYEipk0GEUjLPnYyiGsEPZNCGfJsRQNQOjM9FQDo3BYfGZQj4Ch2JCGsZywf3Fez3vZZsXy9ljzl8Mlhxr3TJuc6yyH37R1faqTZtzXnv395pzTA6qUjwe3LsyVssAnlR3ROierxSf1FVEqtv0w5X3SJKO93jairiKcaZoV5Xx>
I0424 09:51:46.084912    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <8aQft9RcUx6MC7+iQ9IcxuNScJp3cIlm/KhAM4bzKH83zuVZ3eW12mY1Ja8ubTwD8J91nXKmoNrv8fM36u3x0tDQGbJjJi/yuFaxVl1G2w70UdO3YHxPVb7vj6VAMnOk3IB7TEOLiBRsFUdvjRC6SaUyf1gD8Wf2r/6GIicXC4junru/Ni55hL+140TBClfsEbM9mUio>
I0424 09:51:46.084915    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <nKjV0AQz5Hd5REPDZsj29FRP58Ty4IzwkWCKuIt461aUQ9qWg/QKx/iIwxfmzNPQhDPk9rKve2tqcGLHlIXKJWabbin68EcWutqeTQqquDBRR+3necyH7Ig3mkz05E7ym+Rom35MkB9PDhgGHKsBG2Y5Y+1UQJrp7DXp3ZtzVzWjYZsGwCDblPznezfJZ4splYV1dLPz>
I0424 09:51:46.084918    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <WtA18O2+4j3W2Or3ekVtK2YoRuXwqSgDRTkwm4kx2RgD49MZHAZCxegoLGQgbD6bKnxpK+9iAAADtjZnmwWXkiI33273yaC25vewq6zwbCtCohaFoPbV1ReXwhzfmxR+UdH6tPgFBcYuz/XLXZyQsmci4GvV5CXZ66YJUWjG0BkB5ywruD6wMCZaaWZ9urqMxwuL6ZGu>
I0424 09:51:46.084921    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <T3OOec7dUWPyoUOnhiZ6SfZQi9mo3+TU1zevz9pZ6IN2f0MXg2MWseM5O5bAUkulVWPxotNncHezVrV7RQkOZCch8xat5I7daKd6DFpGBGwvaDTs+xdKbgB0/00paSgHRRAalcNC6RyYI6CFhmIYnc9Ro3QqzOYIBRiCslDq6zfUIQuxPlaUdmLJi9bqLLr+voGWkoLX>
I0424 09:51:46.084923    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <buj0Y9bTpHO6UTZ1ueuIhqnn6qYSKz1UvZ0JbbvymL3Z6hs67nyWd0q2ItxtOG5u0BcdUqbLMsppVUnM91Nn00yn+gf/MLGd7jgpJWCrxIdwefTI7cz1JuselGRXW+yE1+KT8hi12f0T8ccsxRnaEQYiuWnEXIVKvprr4fQdO1N5+nuhjradoQqC2iCoFYKaIViutQaW>
I0424 09:51:46.084926    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <g1P5eFAtAoCDlxN1HFJWpIySAQCCOfmzYTJR56OUFYAQTwBBLVyBBWz2ikH6zapCnQUuVA8AnEA1hg2J+ECCrp6ujzhMJBYKJSJxGFUX1tGgRCMw0H+Goq2n64+KpSjFKwKVoDMUkh7o5UmdD1tqJpDeHE8sFhULhBhljUgg2UrxFsaKBMJoqiGsryHg9XT8fYKc1ogi>
I0424 09:51:46.084930    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <UDFsZUqm0WEERqgcOp3BDDYlI9TXpnD8n1WcATz7ZXEkTXFBXj7vw7HqerXVAja6gQBRyzGRgFuA5+MI3rm0ss6QI432uSjxfmu8kh2L8GIejzRGfTP5SBogGOrf1mIO7V52uLNGnCUMOOxUAHYImFVzTw0oS1qrRpbrhtQPuYWPz6uvzC+85t1kIdjItjPtzq7Y+sGd>
I0424 09:51:46.084933    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <4LQhmfveQjlgCssBI/XpOo5EgACciAMAAl7zyU1fISABmkP6VUMnLgwbEKGZBfyvXPw/QeDL7a9DhDchrTch4puQ9psQXgNpPsIriERQ2wzXDK95VTGB4PGf/kn/UZ7Hy9UthNQ1rLpnzQzdOV9V5aUFrdUFDu5rnwRP0nhCn4nyzwMO3e5cLJFuIh25eRzJlu4SzHLf>
I0424 09:51:46.084936    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <CcR91ay9DRkxCJmYP/aEGzrBIrP4SCQ2XLthD0e2gLy8p+1O/br7ZRvKXRJXSosTGnKcfV2rM0XNgf2Z40uT9NK7bpROzOtYvEEhPp+ckOCxiBRvb1F24PHmsoIM0tINOclK5tB2o0NWI7TU+qTeSNKqi+N5pWzV3Bbvo+kHnvpO6vGZvd9eLu+vXI/uMP60cKW13GLU>
I0424 09:51:46.084939    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <5P7zrJ794GmEyzegNvqtDvoMR0QmegcfDmZIoGytjrY8up/1qD2tptRuWOWiYP214fo6h0/KT1msvqDP1H9BMdYPj6SG3SHzT25poU2Tn1/KNEutkfzwYNnzINmwudvAs84vG+s/PpiMo7N45Pj8yYYAgCs5KcusmKx1yy0TLD3d6t4ivNtg7FdYu/Koeazl9emaQx98>
I0424 09:51:46.084941    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: </1CbpUKQncNuRaXK93FxTTn3Fp8ropwwq7Dd/ZWP0pFEDRlKFHnnmJfJbLqXHu5Lkw06+ZKbPLKHv/J0PnW4VzZMKLCrEB3alLqxp2SKG5XSuMJbsc5csJnsuucqfTrd2IF6Yld77bzr6ce1tAyMFwRcK/IMvrJv+0LXyUKdXKaR+GGDW4zR3SPnp+s6k6vC/bh2Npym>
I0424 09:51:46.084947    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <S80ezRQcpui4V+P6uZ6J2SxIzwyCSNFbI4UUKRr2C5kk3gSSDwAA>
I0424 09:51:46.084950    4815 oak_rust_attestation_verifier.cc:207] fcp.foobar: <>
  )text");

  // Extract the serialized log record from the hardcoded logs above.
  std::vector<Record> records;
  int num_records = ExtractRecords(
      hardcoded_record, [&](Record record) { records.push_back(record); });
  EXPECT_EQ(num_records, 0);
  EXPECT_THAT(records, IsEmpty());
}

}  // namespace

}  // namespace fcp::client::attestation
