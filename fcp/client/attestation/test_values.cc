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

#include "fcp/client/attestation/test_values.h"

#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/protos/federatedcompute/confidential_encryption_config.pb.h"
#include "fcp/testing/testing.h"
#include "proto/attestation/reference_value.pb.h"

namespace fcp::client::attestation::test_values {
using ::fcp::confidentialcompute::SignedPayload;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::oak::attestation::v1::ReferenceValues;

ConfidentialEncryptionConfig GetKnownValidEncryptionConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    encryption_key {
      payload: "\010\001\020\001\032\010\315\253\303\202\2010\260\231\" n+dBQh\370\027\360G\001\347\205\315\276cU\257\224\3071\361\337\217\024\226\237\217\353\353\037G"
      signatures {
        headers: "\022\013\010\206\324\270\324\006\020\300\276\310e\032\006\010\211\377\263\324\006\"\006\010\211\351\375\324\006B \2010\260\231\216\2157\351\226\020\237\222G\220\341\265*\274o\037\232\013\301;O\3146\000\320\360#\375\010\002"
        raw_signature: "\361 \361\326\372=H\360!J|\377\363U\322\327-\331\273\036YE6\335\004\277\337b\370\330\227\340)tR\265%\002Y\310!\326\206\320\352,K\217w\027\307<\210\022\244\210\213\335VC9L\261-"
        verifying_key {
          payload: "\010\002\020\004\032\004\217%\276\250\"A\004W{\004\301-\270h\311\200\365\271\357=b\024\376A\341\000\304\311?\243\346c\307`\177\307\277\204\210\375\024\337\265\224\362Xq\336+\332\316{\316A\n\264\006\232\306\255\215t\272 \230\037,\363\355\224\312"
          signatures {
            headers: "\010\002\022\014\010\226\336\202\324\006\020\327\224\322\344\001\032\014\010\342\334\202\324\006\020\327\224\322\344\001\"\014\010\226\310\314\324\006\020\327\224\322\344\001*Chttps://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md2h\n$\010\002: \005,\035y\331\022n\0319\252V\357\306\226\253|d\255\0131\177\007\315c\235\333\2437Tl\266\241\022@\277\264\211\300\212\245t\347^(\224\256M\273\246\373b7\350\336~Te\265\033)n1\010\020\261I\200\016Y}u\227\262V1\006\334}\367g#\010HT@\377y\260\262\361\262l\002\303\335\215-\362"
            log_entry {
              rekor {
                body: "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\"hash\":{\"algorithm\":\"sha256\",\"value\":\"e943c2ce6f74a22b4bdf502301bebd0ebf840f9ead581752ccb1e7145c2fc615\"}},\"signature\":{\"content\":\"MEUCIAbh70Up35V9pGd6YgSYzsgE7YBkw0orcN3BEtlAUTqaAiEAuHYGnRWdcnJyRHfnWTWFwTM8+XfvgR5Xmv/9Jg8gOx4=\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWjh6c1dnbmNrNXdMZHlTQzNBSFpWR2ZHV2F4aAprSllxbXFicTFwcXRzSndTYmU1bEpuZFMwenhvRlBxbU1wdnVpTUx1WWdiSHA1VDlleDl4T1l5RzlBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0tCg==\"}}}}"
                log_index: 2356583091
                tree_size: 2356583276
                hashes: "\335\366\007F\\\200R\264\376\242\336\316E\236\246X`\325a)J\353\206O\260\350\343&\003*\354\214"
                hashes: "\306\335\237\032\246\2057\r\277\246\024\255a\352\033\366#c\020v3\315\261? \333\226\253g\303\321\322"
                hashes: "E\272\344\0061\301\303\351{\027/*o\264\"\315\223\242\264\272>\216\017\322Qp(\036\376\033\205|"
                hashes: "\017\\\343\003\310G\217\302\230\251[\001\217\214\316\007u\205~\204\245[Q\377\022\230\337\273\331\036\"\005"
                hashes: "\035\233\205\301\251~\'][\020\031p_\n\246n\243\376j@A\311\213\034\223\177\016\2272Zp\252"
                hashes: "\370\217{U\237|2\332\275\2170:\271r&)F|\372\270\303\372\271\027m\025\314r\255doQ"
                hashes: "`\004q\267\205\nWC\333\302\037?\345\274\236&\251\336\200\016_\352\365\357\236\222\004\002\000Jl\211"
                hashes: "\263K\r6/s\376\177\377v}\375\331\033\025zu\312\351\327\310yDS\254,\376\202\361?\016\330"
                hashes: "\325\336\341\266\007\343o!#\377\204\314\227\037\367\214i\336\010J\336\324\222\n\325\312\326\266\314\246\231\352"
                hashes: "c\320\353\322zl\200\360O;!\371\327;\030\274\\s\272\346\362J\346\022\2353\343YDf\346\311"
                hashes: "{\336%\r\276\373\\\r\243W\341Y<\254cG\220g\033C\263\\\327H\r)?\306\314\253sJ"
                hashes: "\334\026\353\2613;\207\274\254\251\225\004q\260\377\247#\323\360\316EB\367\000Z\225!\346A\246\004\362"
                hashes: "uJ\210av\301\325\222|\005;P\240\336\3148\177u\301\314e\326\003\374+\324jd\272e\320\025"
                hashes: "g\206\227*U\002\353\231\237\024\375\204K\321Y\277\203\311\201ql\270#\340G\031\014\230\211\314\305j"
                hashes: "\263E\273\233\007\254\242dj%#\373RM@zEX\350\365h\356\327\342\225\373\227\rk\316P\344"
                hashes: "6?\002\020\271\002\026\202T\300\205N\360\247v\305\016Al\023\344\\\353\376$\t\226\202\311A\264\274"
                hashes: "\260\201\001\033VZ\247b\230.\326\033\256\367\263B\004e\320\0147h\227KP@H\r\302\245\024\323"
                hashes: "\363\205\005M\260\006(\244\266XE\024A\233!7\\\275x\375\030\323\001a\325\n\335\360\006\313< "
                hashes: "\324\341\002\320\243\036\221j\365\001\272H\r\205\330\010\345\363\177\246\212\362\026\000+MT\337*f\322+"
                hashes: "\312\004\306\323\313\301\330A&%\003\327F\361y\037N\223\225\247;\330\t\016\014CJ\354\370B>L"
                hashes: "\304\177\303\n\307\213\036\277^*\206\023\362\253\016E\222\273\315WD\031\205\207\271[lV\260\375\347\006"
                checkpoint_origin: "rekor.sigstore.dev - 1193050959916656506"
                checkpoint_signature: "\'ma\005\224\013w\023\322\344\266\301.H\2544\271\0359\025\342\311\330\377I\242\362\222\013\3404\203\373\267\231`\213\\\270|1\310v\335\357\225\r\020\032IDPGhE\010pKE\357\303\343\234\033"
                checkpoint_signature_key_id: "\000\000\000\001"
              }
            }
            verifying_key_id: "\000\000\000\010"
          }
        }
      }
    }
  )pb");
}

SignedPayload GetKnownValidSignedEndorsements() {
  return PARSE_TEXT_PROTO(R"pb(
    payload: "\n \2010\260\231\216\2157\351\226\020\237\222G\220\341\265*\274o\037\232\013\301;O\3146\000\320\360#\375"
    signatures {
      headers: "\010\002\022\013\010\212\324\270\324\006\020\260\202\220L\032\013\010\326\322\270\324\006\020\260\202\220L\"\013\010\212\242\223\330\006\020\260\202\220L"
      raw_signature: "\227\0056*\370K&:\005\001aJ\",j\360k^\374\n\000\307\316\001Ys\334\261U\031\332R\221\'\035\250`1\267>\010\026\300\327h\0226\327\372\353\270\261H\272\200\356\314V\262\035\307\364a\t"
      verifying_key_id: "\000\000\000\014"
    }
  )pb");
}

ReferenceValues GetKnownValidReferenceValues() {
  return PARSE_TEXT_PROTO(R"pb(
    oak_restricted_kernel {
      root_layer {
        amd_sev {
          milan { skip {} }
          stage0 {
            digests {
              digests {
                sha2_384: "\x30\x6c\x92\x15\x73\x7b\xa0\xda\x4c\xc5\xd1\x6b\x65\x5b\x9c\x9d\x66\xfd\x17\xd2\x0c\xba\x47\x22\xbc\x2c\x33\x9b\x1b\x0b\x90\xa6\x1d\xd1\xfe\x78\x1c\x91\xfa\x7a\x1e\xf0\xda\x45\xd1\x19\x5d\x52"
              }
            }
          }
        }
      }
      kernel_layer {
        kernel {
          digests {
            image {
              digests {
                sha2_256: "\x3e\x7c\x37\x18\x58\xf2\xbd\x9c\x03\x28\x94\x69\x4b\xd5\xbb\x48\x93\xf2\x40\x3c\xe8\xa0\xbc\xf7\x3c\x07\xaf\x3e\xf6\xa3\x5a\x15"
              }
            }
            setup_data {
              digests {
                sha2_256: "\x4c\xd0\x20\x82\x0d\xa6\x63\x06\x3f\x41\x85\xca\x14\xa7\xe8\x03\xcd\x7c\x9c\xa1\x48\x3c\x64\xe8\x36\xdb\x84\x06\x04\xb6\xfa\xc1"
              }
            }
          }
        }
        kernel_cmd_line_text { string_literals { value: "console=ttyS0" } }
        init_ram_fs {
          digests {
            digests {
              sha2_256: "\x66\xe7\xb9\xc7\xa2\x49\x83\x6b\xf9\x99\x4a\xfb\x33\x10\x18\x16\x5c\x45\xd7\xfa\x3e\xa8\x27\x96\x02\x2f\xa8\xd4\xd2\x63\xc9\x3e"
            }
          }
        }
        memory_map {
          digests {
            digests {
              sha2_256: "\x3d\x53\x4c\xe7\x94\x90\x22\x88\xc1\x8f\x83\xa9\x93\x0f\xb0\x89\x7b\xd1\xb1\x80\xf7\x1a\xd1\x2d\xef\x2f\x74\xd3\x40\x4f\x21\x32"
            }
          }
        }
        acpi {
          digests {
            digests {
              sha2_256: "\x9a\xfd\x41\x08\xd3\xba\x56\x8a\x2e\x01\x86\x2a\x3c\xb4\x86\x4a\x10\x7e\xc5\x3f\xe7\x3b\x09\x97\x85\xed\x2b\x6e\x63\x1a\xae\x60"
            }
          }
        }
      }
      application_layer {
        binary {
          digests {
            digests {
              sha2_256: "\x2e\x09\x16\x9b\xdb\x55\xd9\x09\x92\x91\x91\xf9\xdd\x52\x3c\x4a\x7f\xb5\x0d\xe0\x10\x94\xa8\x5e\xbd\xf6\xc6\x98\xc9\xb7\x87\x0c"
            }
          }
        }
        # The binary doesn't use any configuration, so nothing to check.
        configuration { skip {} }
      }
    }
  )pb");
}

ReferenceValues GetSkipAllReferenceValues() {
  return PARSE_TEXT_PROTO(R"pb(
    oak_restricted_kernel {
      root_layer {
        amd_sev {
          milan { skip {} }
          genoa { skip {} }
          turin { skip {} }
          stage0 { skip {} }
        }
      }
      kernel_layer {
        kernel { skip {} }
        kernel_cmd_line_text { skip {} }
        init_ram_fs { skip {} }
        memory_map { skip {} }
        acpi { skip {} }
      }
      application_layer {
        binary { skip {} }
        configuration { skip {} }
      }
    }
  )pb");
}

}  // namespace fcp::client::attestation::test_values
