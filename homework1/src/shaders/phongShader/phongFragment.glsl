#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 16
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

// PCF
// #define NEAR_PLANE 8.5
// #define LIGHT_WORLD_SIZE 1.5
// #define LIGHT_FRUSTUM_WIDTH 4.25
// PCSS
#define NEAR_PLANE 9.5
#define LIGHT_WORLD_SIZE 2.
#define LIGHT_FRUSTUM_WIDTH 3.75
// Assuming that LIGHT_FRUSTUM_WIDTH == LIGHT_FRUSTUM_HEIGHT
#define LIGHT_SIZE_UV (LIGHT_WORLD_SIZE / LIGHT_FRUSTUM_WIDTH)

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

uniform int uResolution;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];
vec2 texelSize;

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker(sampler2D shadowMap, vec2 uv, float zReceiver, inout float numBlockers) {

  // TODO:
  float searchWidth = LIGHT_SIZE_UV * (zReceiver - NEAR_PLANE) / zReceiver;
  float blockerSum = 0.0;
  for( int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i ) {
    float shadowMapDepth = unpack(texture2D(shadowMap, uv + poissonDisk[i] * searchWidth * texelSize));
    if ( shadowMapDepth < zReceiver ) {
      blockerSum += shadowMapDepth;
      numBlockers++;
    }
  }
  float avgBlockerDepth = blockerSum / numBlockers;
  return avgBlockerDepth;
}


float PCF(sampler2D shadowMap, vec4 coords) {
  
  // TODO:
  vec2 uv = coords.xy;
  float bias = 0.002;
  float filterRadiusUV = LIGHT_SIZE_UV * NEAR_PLANE / coords.z;
  float sum = 0.0;
  for (int i = 0; i < PCF_NUM_SAMPLES; ++i) {
    vec2 offset = poissonDisk[i] * filterRadiusUV * texelSize;
    float depth = unpack(texture2D(shadowMap, uv + offset));
    sum += depth + bias < coords.z ? 1.0 : 0.0;
  }

  sum /= float(PCF_NUM_SAMPLES);
  float visibility = 1.0 - sum;
  return visibility;
}

// Nearest Neighbor Sample PCF
float PCFSimple(sampler2D shadowMap, vec4 coords) {
  // TODO:
  float bias = 0.006;

  float currentDepth = coords.z;
  float shadow = 0.0;
  for(int x = -1; x <= 1; ++x)
  {
      for(int y = -1; y <= 1; ++y)
      {
          float pcfDepth = texture2D(shadowMap, coords.xy + vec2(x, y) * texelSize).r; 
          shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
      }    
  }
  shadow /= 9.0;
  float visibility = 1.0 - shadow;
  return visibility;
}

float PCSS(sampler2D shadowMap, vec4 coords){

  // TODO:
  vec2 uv = coords.xy;
  float zReceiver = coords.z;
  // STEP 1: avgblocker depth
  float numBlockers = 0.0;
  float avgBlockerDepth = findBlocker(shadowMap, uv, zReceiver, numBlockers);
  if( numBlockers < 1.0 )
    return 1.0;

  // STEP 2: penumbra size

  float zBlocker = avgBlockerDepth;
  float penumbraRatio = (zReceiver - zBlocker) / zBlocker;
  float filterRadiusUV = penumbraRatio * LIGHT_SIZE_UV * NEAR_PLANE / zReceiver;

  // STEP 3: filtering
  float sum = 0.0;
  for (int i = 0; i < PCF_NUM_SAMPLES; ++i) {
    vec2 offset = poissonDisk[i] * filterRadiusUV * texelSize;
    float depth = unpack(texture2D(shadowMap, uv + offset));
    sum += depth < zReceiver ? 1.0 : 0.0;
  }
  return 1.0 - sum / float(PCF_NUM_SAMPLES);
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){

  // TODO:

  // get closest depth value from light's perspective(using [0,1] range vPositionFromLight as coords)
  // 取得最近点的深度(使用[0,1]范围下的 vPositionFromLight 当坐标)
  float closestDepth = unpack(texture2D(shadowMap, shadowCoord.xy));

  // get depth of current fragment from light's perspective
  // 取得当前片段在光源视角下的深度
  float currentDepth = shadowCoord.z;

  // solve shadow acne with a shadow bias 
  // 使用阴影偏移来解决阴影失真问题
  // vec3 lightDir = normalize(uLightPos - vFragPos);
  // float bias = max(0.05 * (1.0 - dot(vNormal, lightDir)), 0.004);
  float bias = 0.000;

  float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

  // 此处返回值应该是可见度(而不是阴影)，1.0是完全可见，0.0完全不可见（指阴影）
  float visibility = 1.0 - shadow;
  return visibility;
  // return 1.0;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {


  float visibility;

  texelSize = 1.0 / vec2(uResolution, uResolution);

  // TODO:
  // perform perspective divide
  // 执行透视除法
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;

  // transform to [0,1] range
  // 转换到 [0,1] 的范围
  shadowCoord = (shadowCoord + 1.0) * 0.5;

  poissonDiskSamples(shadowCoord.xy);
  // uniformDiskSamples(vec2(3378, 35818));

  // visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  // visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  // gl_FragColor = vec4(phongColor, 1.0);
}