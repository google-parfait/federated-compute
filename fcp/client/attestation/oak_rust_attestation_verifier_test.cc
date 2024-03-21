#include "fcp/client/attestation/oak_rust_attestation_verifier.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "fcp/base/digest.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/access_policy.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "fcp/testing/testing.h"
#include "third_party/oak/proto/attestation/endorsement.pb.h"
#include "third_party/oak/proto/attestation/evidence.pb.h"
#include "third_party/oak/proto/attestation/reference_value.pb.h"
#include "third_party/oak/proto/attestation/verification.pb.h"
#include "third_party/oak/proto/digest.pb.h"

namespace fcp::client::attestation {
namespace {
using ::fcp::confidential_compute::OkpCwt;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::oak::attestation::v1::ReferenceValues;
using ::testing::HasSubstr;

// Tests the case where default values are given to the verifier. Verification
// should not succeed in this case, since the Rust Oak Attestation Verification
// library will complain that required values are missing.
//
// This validates that we at least correctly call into the Rust Oak Attestation
// Verification library and get a result back, but it doesn't actually test
// the actual verification logic.
TEST(OakRustAttestationTest, DefaultValuesDoNotVerifySuccessfully) {
  // Generate a new public key, which we'll pass to the client in the
  // ConfidentialEncryptionConfig. We'll use the decryptor from which the public
  // key was generated to validate the encrypted payload at the end of the test.
  fcp::confidential_compute::MessageDecryptor decryptor;
  auto encoded_public_key =
      decryptor
          .GetPublicKey(
              [](absl::string_view payload) { return "fakesignature"; }, 0)
          .value();
  absl::StatusOr<OkpCwt> parsed_public_key = OkpCwt::Decode(encoded_public_key);
  ASSERT_OK(parsed_public_key);
  ASSERT_TRUE(parsed_public_key->public_key.has_value());

  // Note: we don't specify any attestation evidence nor attestation
  // endorsements in the encryption config, since we can't generate valid
  // attestations in a test anyway.
  ConfidentialEncryptionConfig encryption_config;
  encryption_config.set_public_key(encoded_public_key);
  // Populate an empty Evidence proto.
  encryption_config.mutable_attestation_evidence();

  // Use an empty ReferenceValues input.
  ReferenceValues reference_values;
  OakRustAttestationVerifier verifier(reference_values, {});

  // The verification should fail, since neither reference values nor the
  // evidence are valid.
  auto result = verifier.Verify(absl::Cord(""), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

ConfidentialEncryptionConfig GetKnownValidEncryptionConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    public_key: "\204C\241\001&\240XH\243\004\032f\000\255\364\006\032e\367st:\000\001\000\000X4\245\001\001\002D6\235\225b\003:\000\001\000\000 \004!X +\235%K\364\321e\215\314\265\331\306]\304\027\305\244\263\352\021\210\240\371y\026#>\346\0347\\]X@\322\370\026\260\277/\252\204+h\241V\275\031\237\010\347n0r\274\020dM\212\177\275\260\357\335\215\351Qz\271{e\332\213\306\221\037\372\351*\016\r\225r\3376\000\340+\376*\266\312\315KL\002\020;"
    attestation_evidence {
      root_layer {
        platform: AMD_SEV_SNP
        remote_attestation_report: "\002\000\000\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\003\000\000\000\000\000\024\321\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\216*];\303\213\243v\037@\327\302Tpd\361g\260xwv`i\317_\337%\375\\\256\355\346\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\366\337 T\243\207\363\370)\221A\226\010mY\222dk|\275\203Bp\304\333 \\\323hy\227~\340`\026\301\346\\\236\304S\3434\2415>\223:\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\311r\000\362\t\312\202\203\304k*\204\244\0328y=\026\321 \213\274\372\036\220.\346\237\362\227\370\017\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\003\000\000\000\000\000\024\321\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\343\014\".\276b\337\n\nq\203\205Z\352\003\330\315\344\210P8\214\002\274\306\347&^\223\343(\211\330\236\326\360\nny/\340gy\270\260\354@NF\205\300ADkP{\237\370U\325>U\0071\003\000\000\000\000\000\024\321\0207\001\000\0207\001\000\003\000\000\000\000\000\024\321\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\356R\303|2\266\337ql\210\320\236|\311\233@\036:\267t+A\306\231\372\264jmI\230\353\031V$\317roY\366\374\036\204\226D\270\322\344\037\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\277[\332\2019i\323\207\232\337Np\212W\305C\024\352\300SC\216\233\315wj\200\336T\251D\\\252\001\022$\232\010\003\203\246j\334\205\010!Y\216\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
        eca_public_key: "\247\001\002\002T\200\260\316\243\\\222M\375)v\231GH\201\006\216\262\034\353I\003&\004\201\002 \001!X |+~1\003\025\017\234p\'\007\034\231W6\010\276\031\311\027}\357\300\323\222\255\317\332\274\007;\323\"X \317\313k\316\"\275\263\360^2a\376\211\260{\037#\275g\031\007\213p\223\010\237\224\325j\315\341\334"
      }
      layers {
        eca_certificate: "\204C\241\001&\241\004RAsymmetricECDSA256Y\001\340\245\001x(80b0cea35c924dfd297699474881068eb21ceb49\002x(443fd3026ba923e5ac6c779dedbce0bcb65efbd9:\000GDWXf\247\001\002\002TD?\323\002k\251#\345\254lw\235\355\274\340\274\266^\373\331\003&\004\201\002 \001!X \234\023_Tv\353\235;\262.V@\0340\235\'\347\370\314\035A(\346\017B\246\350\257\351j\306\213\"X \256\257\030<tl\262t\334J\300\303\314\206snR\351\010P&\000{\271\261\316\311?E\001\342D:\000GDXB \000:\000GDZ\246:\000GD`\241:\000GDkX \314\216\243\312j\305\340\247s\342[\037\017}\365j\356\340wB\033R\206\375\345BOc\005\007\373N:\000GDa\241:\000GDkX +\230Xm\231\005\246\005\302\225\327|a\350\317\322\002z\345\270\240N\357\251\001\2046\366\255\021B\227:\000GDb\241:\000GDkX L\320 \202\r\246c\006?A\205\312\024\247\350\003\315|\234\241H<d\3506\333\204\006\004\266\372\301:\000GDc\241:\000GDkX \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000:\000GDd\241:\000GDkX [k\321d\237\236\233t.\001\316\246\261E\375#\3110\252\212\246\t\365\255QS\377sZ3qe:\000GDe\241:\000GDkX \333\254\312\347\277\277\000n+\206#\250/\032_\315\242\352\003\222#<&\261\203V\263\274\372\3021\353X@[\331\256\251j}\263\207\250M\330\347pL\243\202\372|\257\336\315F\\\335\205\321\306P\275\315+\332\034\035U\014\263|\311\354x\"\331H\371X^\315\317\326\273I|k\224\006\t\000\030\374\330\313\361Z"
      }
      application_keys {
        encryption_public_key_certificate: "\204C\241\001&\241\004RAsymmetricECDSA256X\351\245\001x(443fd3026ba923e5ac6c779dedbce0bcb65efbd9\002x(ae3c5167d448942425ff7fae990a2157a14ab168:\000GDWXD\246\001\001\002T\256<Qg\324H\224$%\377\177\256\231\n!W\241J\261h\0038\036\004\201\005 \004!X \347\253\020\227\317X\027\362\232;\350\271\326\376{\371Y\240\245\030\231\207\245\351k\314x\016\035f\203\':\000GDXB \000:\000GD[\242:\000GDf\241:\000GDkX s\333\35045_R\317\244\264!\247\303\325\004dc*\341\360\231V\007\316\265;iq\310BUU:\000GDi\241:\000GDk@X@H\375\252\247Ton\301\272\227\347 \021\324\377\261\t\233\337w\321\335\032\237\354@\003hp\036\322k\324D\206\016l\3662C\014\360\'\001_\0204\276\300\360\2142\266\274\333\337g\002.$\006\217[\025"
        signing_public_key_certificate: "\204C\241\001&\241\004RAsymmetricECDSA256Y\001\013\245\001x(443fd3026ba923e5ac6c779dedbce0bcb65efbd9\002x(c3a6211254e6e4de9ed43f0e49b5a46245cd8d52:\000GDWXf\247\001\002\002T\303\246!\022T\346\344\336\236\324?\016I\265\244bE\315\215R\003&\004\201\002 \001!X ) pV\021@U\371NF\261NU\346\343\361\360\210\276\243\273\310\307\323\314\024\001aU\035B\261\"X 9s\225\034\210\347c\215\200\003\225\031\035\261\372\022QZ\027B5\314)\034\252\030\236\262\264\264\336\031:\000GDXB \000:\000GD[\242:\000GDf\241:\000GDkX s\333\35045_R\317\244\264!\247\303\325\004dc*\341\360\231V\007\316\265;iq\310BUU:\000GDi\241:\000GDk@X@IK\356\030\300E6\240\277}\311\213s\242\206r\220L\035\375\221\317\031%\237\276\362\3376\352rJ\tR,\271\333\210\217\205\315\007\033~U/\303iUO\263\377z9\036^ci\030\317\373RM>"
      }
    }
    attestation_endorsements {
      oak_restricted_kernel {
        root_layer {
          tee_certificate: "0\202\005M0\202\002\374\240\003\002\001\002\002\001\0000F\006\t*\206H\206\367\r\001\001\n09\240\0170\r\006\t`\206H\001e\003\004\002\002\005\000\241\0340\032\006\t*\206H\206\367\r\001\001\0100\r\006\t`\206H\001e\003\004\002\002\005\000\242\003\002\0010\243\003\002\001\0010{1\0240\022\006\003U\004\013\014\013Engineering1\0130\t\006\003U\004\006\023\002US1\0240\022\006\003U\004\007\014\013Santa Clara1\0130\t\006\003U\004\010\014\002CA1\0370\035\006\003U\004\n\014\026Advanced Micro Devices1\0220\020\006\003U\004\003\014\tSEV-Milan0\036\027\r240303194456Z\027\r310303194456Z0z1\0240\022\006\003U\004\013\014\013Engineering1\0130\t\006\003U\004\006\023\002US1\0240\022\006\003U\004\007\014\013Santa Clara1\0130\t\006\003U\004\010\014\002CA1\0370\035\006\003U\004\n\014\026Advanced Micro Devices1\0210\017\006\003U\004\003\014\010SEV-VCEK0v0\020\006\007*\206H\316=\002\001\006\005+\201\004\000\"\003b\000\004D\2362\254\336[\222\316(\236a\006\337\324z\2508v3Xu\351\366\316q\320\373\350tTc\363\353p\323\033\n\205>\221\336\277n\232eT\221-\241\002\335c6\270\030\252\354\250\247\324\270\366F\013[\306\363=k\371\031\306\266\240\243\307D\317\034c^8\'\027\343\233\263oO\326Z\220\363y?\210\243\202\001\0270\202\001\0230\020\006\t+\006\001\004\001\234x\001\001\004\003\002\001\0000\027\006\t+\006\001\004\001\234x\001\002\004\n\026\010Milan-B00\021\006\n+\006\001\004\001\234x\001\003\001\004\003\002\001\0030\021\006\n+\006\001\004\001\234x\001\003\002\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\004\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\005\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\006\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\007\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\003\004\003\002\001\0240\022\006\n+\006\001\004\001\234x\001\003\010\004\004\002\002\000\3210M\006\t+\006\001\004\001\234x\001\004\004@\343\014\".\276b\337\n\nq\203\205Z\352\003\330\315\344\210P8\214\002\274\306\347&^\223\343(\211\330\236\326\360\nny/\340gy\270\260\354@NF\205\300ADkP{\237\370U\325>U\00710F\006\t*\206H\206\367\r\001\001\n09\240\0170\r\006\t`\206H\001e\003\004\002\002\005\000\241\0340\032\006\t*\206H\206\367\r\001\001\0100\r\006\t`\206H\001e\003\004\002\002\005\000\242\003\002\0010\243\003\002\001\001\003\202\002\001\000[\302\311J\331!6f~\037\332\270C\005\nP\013\001\215\211\333\373\002\2452XeE\372\255\256T\216\325\336,rt_\010\222\314\2313\233t\200c\016?~\001x\305\321\007p3\366\021[\372\037\370\356Bf\3727\r7b3md\364\301Y\2079y\"\rF\345\327\326\306Q\337\254Y\255:\201It\246{\311\235.N<\271\230i\321U\353\230\371;\203\014\224\342\313\251\372\031\335,Y\242n\267\204{{@)\010z(\034\254\214\351h\254\241\225\010;Y\235\204\2506\363w\023\216\036\3662\213\306\203\347m\010J\275\371\237\2518\332\030\323D\226\224\214\361N\373\014b6\347\322\300\255\353\266Wa|\024\223\243I!\202\034\367\025\337\365\232\345\212\002\2573Bb\0211\316RLV\253\000\0063\372\347\355\344\355\225r\n\233\005\335\327\2374R!\367(2\277\251\'\364\332:\2427\217\311\310Q*\177\255\252\034L\274\0206\020\375 \024\020km1g\326\rb\245j\3232\376\r\365\276\230\032\213\277r\346\340>\365Vy\364\033=\354\362\336\265\316\306}\215\204\00047X\rz\240\373\311T\001Br\245y\230\263\373\301=\236\254c;\257\324?\323e\330\311\024R\243\301I\226\033v\035\310\376\277\216O\341\344\0077\33233~\364=\244\251\250K\000x\317\235\331,\264\244 \234\032\263&\205\305E\250+\0101[\363\201iD\235\033\254y%\343;\221\352\220y\355-N\r\317@\233\364\305A.\252\221\347y\364\004\241\'\263i\216_\213^\345\247\374Bq\206\316HD\242Q\033ch\r<\207\3044\376\224\024*1\234\200\333\301\031\310\224\231\005\005\021\024\"T\307\244AZ\303\211w#<\373\243\t\2366\023n\344\311=u\023\330\222\267\376\302\336\204\270kRB\'%9\317\276\321@\321 \000d\242\335\331\277<\256\014\025"
        }
      }
    }
  )pb");
}

ReferenceValues GetKnownValidReferenceValues() {
  return PARSE_TEXT_PROTO(R"pb(
    oak_restricted_kernel {
      # This verifies the AMD-SEV root layer, but skips the stage0 validation
      # since that isn't supported yet (see http://b/327069120)
      root_layer { amd_sev { stage0 { skip {} } } }
      kernel_layer {
        kernel {
          digests {
            image {
              digests {
                sha2_256: "\xcc\x8e\xa3\xca\x6a\xc5\xe0\xa7\x73\xe2\x5b\x1f\x0f\x7d\xf5\x6a\xee\xe0\x77\x42\x1b\x52\x86\xfd\xe5\x42\x4f\x63\x05\x07\xfb\x4e"
              }
            }
            setup_data {
              digests {
                sha2_256: "\x4c\xd0\x20\x82\x0d\xa6\x63\x06\x3f\x41\x85\xca\x14\xa7\xe8\x03\xcd\x7c\x9c\xa1\x48\x3c\x64\xe8\x36\xdb\x84\x06\x04\xb6\xfa\xc1"
              }
            }
          }
        }
        kernel_cmd_line {
          digests {
            digests {
              sha2_256: "\x2b\x98\x58\x6d\x99\x05\xa6\x05\xc2\x95\xd7\x7c\x61\xe8\xcf\xd2\x02\x7a\xe5\xb8\xa0\x4e\xef\xa9\x01\x84\x36\xf6\xad\x11\x42\x97"
            }
          }
        }
        init_ram_fs {
          digests {
            digests {
              sha2_256: "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            }
          }
        }
        memory_map {
          digests {
            digests {
              sha2_256: "\x5b\x6b\xd1\x64\x9f\x9e\x9b\x74\x2e\x01\xce\xa6\xb1\x45\xfd\x23\xc9\x30\xaa\x8a\xa6\x09\xf5\xad\x51\x53\xff\x73\x5a\x33\x71\x65"
            }
          }
        }
        acpi {
          digests {
            digests {
              sha2_256: "\xdb\xac\xca\xe7\xbf\xbf\x00\x6e\x2b\x86\x23\xa8\x2f\x1a\x5f\xcd\xa2\xea\x03\x92\x23\x3c\x26\xb1\x83\x56\xb3\xbc\xfa\xc2\x31\xeb"
            }
          }
        }
      }
      application_layer {
        binary {
          digests {
            digests {
              sha2_256: "\x73\xdb\xe8\x34\x35\x5f\x52\xcf\xa4\xb4\x21\xa7\xc3\xd5\x04\x64\x63\x2a\xe1\xf0\x99\x56\x07\xce\xb5\x3b\x69\x71\xc8\x42\x55\x55"
            }
          }
        }
        # The binary doesn't use any configuration, so nothing to check.
        configuration { skip {} }
      }
    }
  )pb");
}

TEST(OakRustAttestationTest,
     KnownValidEncryptionConfigAndValidPolicyInAllowlist) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "foo" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will only accept attestation evidence matching the reference
  // values defined above, and will only accept the given access policy.
  OakRustAttestationVerifier verifier(
      reference_values,
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))});

  // Ensure that the verification succeeds.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  ASSERT_OK(result);
}

TEST(OakRustAttestationTest, KnownValidEncryptionConfigAndMismatchingPolicy) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy disallowed_access_policy =
      PARSE_TEXT_PROTO(R"pb(
        transforms {
          src: 0
          application { tag: "bar" }
        }
      )pb");
  auto disallowed_access_policy_bytes =
      disallowed_access_policy.SerializeAsString();

  // This verifier will not accept any inputs, since the policy allowlist
  // doesn't match the actual policy.
  OakRustAttestationVerifier verifier(reference_values,
                                      {"mismatching policy hash"});

  // Ensure that the verification *does not* succeed.
  auto result = verifier.Verify(absl::Cord(disallowed_access_policy_bytes),
                                encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Data access policy not in allowlist"));
}

TEST(OakRustAttestationTest,
     KnownValidEncryptionConfigAndAndValidPolicyWithEmptyPolicyAllowlist) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "foo" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will not accept any inputs, since the policy allowlist is
  // empty.
  OakRustAttestationVerifier verifier(reference_values, {});

  // Ensure that the verification *does not* succeed.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Data access policy not in allowlist"));
}

TEST(OakRustAttestationTest,
     KnownValidEncryptionConfigAndAndEmptyPolicyWithEmptyPolicyAllowlist) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // This verifier will not accept any inputs, since the policy allowlist is
  // empty.
  OakRustAttestationVerifier verifier(reference_values, {});

  // Ensure that the verification *does not* succeed, since an empty access
  // policy string still has to match an allowlist entry.
  auto result = verifier.Verify(absl::Cord(""), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Data access policy not in allowlist"));
}

TEST(OakRustAttestationTest,
     KnownEncryptionConfigAndMismatchingReferencevalues) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();
  // Mess with the application layer digest value to ensure it won't match the
  // values in the ConfidentialEncryptionConfig.
  (*reference_values.mutable_oak_restricted_kernel()
        ->mutable_application_layer()
        ->mutable_binary()
        ->mutable_digests()
        ->mutable_digests(0)
        ->mutable_sha2_256())[0] += 1;

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "bar" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will not accept the encryption config provided, due to the
  // mismatching digest.
  OakRustAttestationVerifier verifier(
      reference_values,
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))});

  // Ensure that the verification *does not* succeed.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

TEST(OakRustAttestationTest, KnownEncryptionConfigAndEmptyReferencevalues) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "bar" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will not accept the encryption config provided, due to the
  // reference values being invalid (an empty, uninitialized proto).
  OakRustAttestationVerifier verifier(
      ReferenceValues(),
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))});

  // Ensure that the verification *does not* succeed.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

}  // namespace

}  // namespace fcp::client::attestation
