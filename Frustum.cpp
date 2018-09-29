#include<glm/gtc/matrix_access.hpp>
#include "Frustum.h"
//#include "Plane.h"

using namespace glm;

Frustum::Frustum(){
    lines.resize(24);
    glGenVertexArrays(1,&frustum);
    glBindVertexArray(frustum);
    glGenBuffers(1,&frustumBuffer);

    drawFrustum = (std::unique_ptr<ShaderProgram>(new ShaderProgram()));
    drawFrustum->initFromFiles("shaders/drawFrustum.vert", "shaders/drawFrustum.frag");
    drawFrustum->addUniform("VP");
    drawFrustum->addUniform("frustumColor");
    drawFrustum->addAttribute("position");
    glBindVertexArray(0);
}


Frustum::~Frustum(){
    glDeleteBuffers(1, &frustumBuffer);
    glDeleteVertexArrays(1, &frustum);
}

//This function should be enough for our use
void Frustum::setFromVectors(const vec3& dir, const vec3& pos, const vec3& right,
        const vec3& up, float Near, float Far, float fov, float aspect) {

    float angleTangent = tan(fov / 2);
    float heightFar = angleTangent * Far;
    float widthFar = heightFar * aspect;
    float heightNear = angleTangent * Near;
    float widthNear = heightNear * aspect;
    vec3 farCenter = pos + dir * Far;
    vec3 farTopLeft = farCenter + (up * heightFar) - (right* widthFar);
    vec3 farTopRight = farCenter + (up * heightFar) + (right* widthFar);
    vec3 farBotLeft = farCenter - (up * heightFar) - (right* widthFar);
    vec3 farBotRight = farCenter - (up * heightFar) + (right* widthFar);

    vec3 nearCenter = pos + dir * Near;
    vec3 nearTopLeft = nearCenter + (up * heightNear) - (right* widthNear);
    vec3 nearTopRight = nearCenter + (up * heightNear) + (right* widthNear);
    vec3 nearBotLeft = nearCenter - (up * heightNear) - (right* widthNear);
    vec3 nearBotRight = nearCenter - (up * heightNear) + (right* widthNear);

    /*
    near = Plane(nearBotLeft, nearTopLeft, nearBotRight);
    far = Plane(farTopRight, farTopLeft, farBotRight);
    left = Plane(farTopLeft, nearTopLeft, farBotLeft);
    right = Plane(nearTopRight, farTopRight, nearBotRight);
    top = Plane(nearTopLeft, farTopLeft, nearTopRight);
    bottom = Plane(nearBotRight, farBotLeft, nearBotLeft);
    */

/*
    corners[0] = farTopLeft;
    corners[1] = farTopRight; 
    corners[2] = farBotLeft; 
    corners[3] = farBotRight; 
    corners[4] = nearBotRight; 
    corners[5] = nearTopLeft; 
    corners[6] = nearTopRight; 
    corners[7] = nearBotLeft; 
*/
    corners[0] = vec3(-1,1,0);
    corners[1] = vec3(1,1,0); 
    corners[2] = vec3(-1,-1,0); 
    corners[3] = vec3(1,-1,0); 
    corners[4] = vec3(1,-1,-1); 
    corners[5] = vec3(-1,1,-1);
    corners[6] = vec3(1,1,-1); 
    corners[7] = vec3(-1,-1,-1); 

    // Far face lines.
    lines[0] = corners[0];
    lines[1] = corners[1];
    lines[2] = corners[3];
    lines[3] = corners[2];
    lines[4] = corners[1];
    lines[5] = corners[3];
    lines[6] = corners[2];
    lines[7] = corners[0];

    // Near face lines.
    lines[8] = corners[4];
    lines[9] = corners[7];
    lines[10] = corners[6];
    lines[11] = corners[5];
    lines[12] = corners[5];
    lines[13] = corners[7];
    lines[14] = corners[6];
    lines[15] = corners[4];

    // Connecting lines.
    lines[16] = corners[0];
    lines[17] = corners[5];
    lines[18] = corners[1];
    lines[19] = corners[6];
    lines[20] = corners[2];
    lines[21] = corners[7];
    lines[22] = corners[3];
    lines[23] = corners[4];

    uploadBuffer();
} 
    
void Frustum::uploadBuffer() {
    glBindVertexArray(frustum);
    glBindBuffer(GL_ARRAY_BUFFER, frustumBuffer);
    glBufferData(GL_ARRAY_BUFFER, 24*sizeof(glm::vec3), lines.data(), GL_DYNAMIC_DRAW );

    glEnableVertexAttribArray(drawFrustum->attribute("position"));
    glVertexAttribPointer(drawFrustum->attribute("position"), 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindVertexArray(0);
}

void Frustum::draw(const glm::mat4& transform){
    mat4 VP = glm::inverse(transform);//mat4(1);//transform;//
    //const mat4 &VP = transform;
    drawFrustum->use();
    glBindVertexArray(frustum);
    glUniformMatrix4fv(drawFrustum->uniform("VP"), 1, false, glm::value_ptr(VP));
    glUniform3f(drawFrustum->uniform("frustumColor"), 1,1,1);
    glDrawArrays(GL_LINES, 0, 24);
    //glDrawArrays(GL_POINTS, 0, 24);
    glBindVertexArray(0);

}

/*
void Frustum::setFromViewProj(cosnt mat4& view, cosnt mat4& proj) {
    vec3 right = transpose(view)
}
*/

/* TODO 
 * Check correctness of following function
 * 
void Frustum::setFromParams(const mat4& view, float near, float far, float fx,
                float fy, float cx, float cy, float imgWidth, float imgHeight) {

    vec3 right = vec3(glm::column(view, 0));
    vec3 up = -vec3(glm::column(view, 1));
    vec3 d = vec3(glm::column(view, 2));

    //mat3 r = view.linearrVeco();
//    Vec3 right = r.col(0);
//    Vec3 up = -r.col(1);
//    Vec3 d = r.col(2);
    vec3 p = vec3(view[3]);
//    Vec3 p = view.translation();
    float aspect = (fx * imgWidth) / (fy * imgHeight);
    float fov = atan2(cy, fy) + atan2(imgHeight - cy, fy);
    setFromVectors(d, p, right, up, near, far, fov, aspect);
}
*/

